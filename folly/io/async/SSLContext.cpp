/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <folly/io/async/SSLContext.h>

#include <folly/Format.h>
#include <folly/Memory.h>
#include <folly/Random.h>
#include <folly/SharedMutex.h>
#include <folly/ssl/OpenSSLTicketHandler.h>
#include <folly/ssl/PasswordCollector.h>
#include <folly/ssl/SSLSessionManager.h>

// ---------------------------------------------------------------------
// SSLContext implementation
// ---------------------------------------------------------------------
namespace folly {

namespace {

int getExDataIndex() {
  static auto index =
      SSL_CTX_get_ex_new_index(0, nullptr, nullptr, nullptr, nullptr);
  return index;
}

/**
 * Configure the given SSL context to use the given version.
 */
void configureProtocolVersion(SSL_CTX* ctx, SSLContext::SSLVersion version) {
  /*
   * From the OpenSSL docs https://fburl.com/ii9k29qw:
   * Setting the minimum or maximum version to 0, will enable protocol versions
   * down to the lowest version, or up to the highest version supported by the
   * library, respectively.
   *
   * We can use that as the default/fallback.
   */
  int minVersion = 0;
  switch (version) {
    case SSLContext::SSLVersion::TLSv1:
      minVersion = TLS1_VERSION;
      break;
    case SSLContext::SSLVersion::SSLv3:
      minVersion = SSL3_VERSION;
      break;
    case SSLContext::SSLVersion::TLSv1_2:
      minVersion = TLS1_2_VERSION;
      break;
    case SSLContext::SSLVersion::TLSv1_3:
      minVersion = TLS1_3_VERSION;
      break;
    case SSLContext::SSLVersion::SSLv2:
    default:
      // do nothing
      break;
  }
  const auto setMinProtoResult = SSL_CTX_set_min_proto_version(ctx, minVersion);
  DCHECK(setMinProtoResult == 1)
      << sformat("unsupported min TLS protocol version: 0x{:04x}", minVersion);
}

static int dispatchTicketCrypto(
    SSL* ssl,
    unsigned char* keyName,
    unsigned char* iv,
    EVP_CIPHER_CTX* cipherCtx,
    HMAC_CTX* hmacCtx,
    int encrypt) {
  auto ctx = folly::SSLContext::getFromSSLCtx(SSL_get_SSL_CTX(ssl));
  DCHECK(ctx);

  auto handler = ctx->getTicketHandler();
  if (!handler) {
    LOG(FATAL) << "Null OpenSSLTicketHandler in callback";
  }

  return handler->ticketCallback(ssl, keyName, iv, cipherCtx, hmacCtx, encrypt);
}
} // namespace

//
// For OpenSSL portability API

// SSLContext implementation
SSLContext::SSLContext(SSLVersion version) {
  ctx_ = SSL_CTX_new(TLS_method());
  if (ctx_ == nullptr) {
    throw std::runtime_error("SSL_CTX_new: " + getErrors());
  }

  // configure the TLS version used
  configureProtocolVersion(ctx_, version);

  SSL_CTX_set_mode(ctx_, SSL_MODE_AUTO_RETRY);

  checkPeerName_ = false;

  SSL_CTX_set_options(ctx_, SSL_OP_NO_COMPRESSION);

  sslAcceptRunner_ = std::make_unique<SSLAcceptRunner>();

  setupCtx(ctx_);

  SSL_CTX_set_tlsext_servername_callback(ctx_, baseServerNameOpenSSLCallback);
  SSL_CTX_set_tlsext_servername_arg(ctx_, this);
}

SSLContext::~SSLContext() {
  if (ctx_ != nullptr) {
    SSL_CTX_free(ctx_);
    ctx_ = nullptr;
  }

  deleteNextProtocolsStrings();
}

void SSLContext::ciphers(const std::string& ciphers) {
  setCiphersOrThrow(ciphers);
}

void SSLContext::setClientECCurvesList(
    const std::vector<std::string>& ecCurves) {
  if (ecCurves.empty()) {
    return;
  }
  std::string ecCurvesList;
  join(":", ecCurves, ecCurvesList);
  const auto rc = SSL_CTX_set1_curves_list(ctx_, ecCurvesList.c_str());
  if (rc == 0) {
    throw std::runtime_error("SSL_CTX_set1_curves_list " + getErrors());
  }
}

void SSLContext::setSupportedGroups(const std::vector<std::string>& groups) {
  if (groups.empty()) {
    return;
  }
  std::string groupsList;
  join(":", groups, groupsList);
  const auto rc = SSL_CTX_set1_groups_list(ctx_, groupsList.c_str());
  if (rc == 0) {
    throw std::runtime_error("SSL_CTX_set1_curves " + getErrors());
  }
}

void SSLContext::setServerECCurve(const std::string& curveName) {
  EC_KEY* ecdh = nullptr;
  int nid;

  /*
   * Elliptic-Curve Diffie-Hellman parameters are either "named curves"
   * from RFC 4492 section 5.1.1, or explicitly described curves over
   * binary fields. OpenSSL only supports the "named curves", which provide
   * maximum interoperability.
   */

  nid = OBJ_sn2nid(curveName.c_str());
  if (nid == 0) {
    LOG(FATAL) << "Unknown curve name:" << curveName.c_str();
  }
  ecdh = EC_KEY_new_by_curve_name(nid);
  if (ecdh == nullptr) {
    LOG(FATAL) << "Unable to create curve:" << curveName.c_str();
  }

  SSL_CTX_set_tmp_ecdh(ctx_, ecdh);
  EC_KEY_free(ecdh);
}

SSLContext::SSLContext(SSL_CTX* ctx) : ctx_(ctx) {
  setupCtx(ctx);
  if (SSL_CTX_up_ref(ctx) == 0) {
    throw std::runtime_error("Failed to increment SSL_CTX refcount");
  }
}

void SSLContext::setX509VerifyParam(
    const ssl::X509VerifyParam& x509VerifyParam) {
  if (!x509VerifyParam) {
    return;
  }
  if (SSL_CTX_set1_param(ctx_, x509VerifyParam.get()) != 1) {
    throw std::runtime_error("SSL_CTX_set1_param " + getErrors());
  }
}

void SSLContext::setCiphersOrThrow(const std::string& ciphers) {
  const auto rc = SSL_CTX_set_cipher_list(ctx_, ciphers.c_str());
  if (rc == 0) {
    throw std::runtime_error("SSL_CTX_set_cipher_list: " + getErrors());
  }
  providedCiphersString_ = ciphers;
}

void SSLContext::setSigAlgsOrThrow(const std::string& sigalgs) {
  const auto rc = SSL_CTX_set1_sigalgs_list(ctx_, sigalgs.c_str());
  if (rc == 0) {
    throw std::runtime_error("SSL_CTX_set1_sigalgs_list " + getErrors());
  }
}

void SSLContext::setVerificationOption(
    const SSLContext::SSLVerifyPeerEnum& verifyPeer) {
  CHECK(verifyPeer != SSLVerifyPeerEnum::USE_CTX); // dont recurse
  verifyPeer_ = verifyPeer;
}

void SSLContext::setVerificationOption(
    const SSLContext::VerifyClientCertificate& verifyClient) {
  verifyClient_ = verifyClient;
}

void SSLContext::setVerificationOption(
    const SSLContext::VerifyServerCertificate& verifyServer) {
  verifyServer_ = verifyServer;
}

int SSLContext::getVerificationMode(
    const SSLContext::VerifyClientCertificate& verifyClient) {
  int mode = SSL_VERIFY_NONE;
  switch (verifyClient) {
    case VerifyClientCertificate::ALWAYS:
      mode = SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT;
      break;

    case VerifyClientCertificate::IF_PRESENTED:
      mode = SSL_VERIFY_PEER;
      break;

    case VerifyClientCertificate::DO_NOT_REQUEST:
      mode = SSL_VERIFY_NONE;
      break;
  }
  return mode;
}

int SSLContext::getVerificationMode(
    const SSLContext::VerifyServerCertificate& verifyServer) {
  int mode = SSL_VERIFY_NONE;
  switch (verifyServer) {
    case VerifyServerCertificate::IF_PRESENTED:
      mode = SSL_VERIFY_PEER;
      break;

    case VerifyServerCertificate::IGNORE_VERIFY_RESULT:
      mode = SSL_VERIFY_NONE;
      break;
  }
  return mode;
}

int SSLContext::getVerificationMode(
    const SSLContext::SSLVerifyPeerEnum& verifyPeer) {
  CHECK(verifyPeer != SSLVerifyPeerEnum::USE_CTX);
  int mode = SSL_VERIFY_NONE;
  switch (verifyPeer) {
      // case SSLVerifyPeerEnum::USE_CTX: // can't happen
      // break;

    case SSLVerifyPeerEnum::VERIFY:
      mode = SSL_VERIFY_PEER;
      break;

    case SSLVerifyPeerEnum::VERIFY_REQ_CLIENT_CERT:
      mode = SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT;
      break;

    case SSLVerifyPeerEnum::NO_VERIFY:
      mode = SSL_VERIFY_NONE;
      break;
    case SSLVerifyPeerEnum::USE_CTX:
    default:
      break;
  }
  return mode;
}

int SSLContext::getVerificationMode() const {
  // the below or'ing is incorrect unless VERIFY_NONE is 0
  static_assert(SSL_VERIFY_NONE == 0);
  return getVerificationMode(verifyClient_) |
      getVerificationMode(verifyServer_) | getVerificationMode(verifyPeer_);
}

void SSLContext::authenticate(
    bool checkPeerCert, bool checkPeerName, const std::string& peerName) {
  int mode;
  if (checkPeerCert) {
    mode = SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT |
        SSL_VERIFY_CLIENT_ONCE;
    checkPeerName_ = checkPeerName;
    peerFixedName_ = peerName;
  } else {
    mode = SSL_VERIFY_NONE;
    checkPeerName_ = false; // can't check name without cert!
    peerFixedName_.clear();
  }
  SSL_CTX_set_verify(ctx_, mode, nullptr);
}

void SSLContext::loadCertificate(const char* path, const char* format) {
  if (path == nullptr || format == nullptr) {
    throw std::invalid_argument(
        "loadCertificateChain: either <path> or <format> is nullptr");
  }
  if (strcmp(format, "PEM") == 0) {
    if (SSL_CTX_use_certificate_chain_file(ctx_, path) != 1) {
      int errnoCopy = errno;
      std::string reason("SSL_CTX_use_certificate_chain_file: ");
      reason.append(path);
      reason.append(": ");
      reason.append(getErrors(errnoCopy));
      throw std::runtime_error(reason);
    }
  } else {
    throw std::runtime_error(
        "Unsupported certificate format: " + std::string(format));
  }
}

void SSLContext::loadCertificateFromBufferPEM(folly::StringPiece cert) {
  if (cert.data() == nullptr) {
    throw std::invalid_argument("loadCertificate: <cert> is nullptr");
  }

  ssl::BioUniquePtr bio(BIO_new(BIO_s_mem()));
  if (bio == nullptr) {
    throw std::runtime_error("BIO_new: " + getErrors());
  }

  int written = BIO_write(bio.get(), cert.data(), int(cert.size()));
  if (written <= 0 || static_cast<unsigned>(written) != cert.size()) {
    throw std::runtime_error("BIO_write: " + getErrors());
  }

  ssl::X509UniquePtr x509(
      PEM_read_bio_X509(bio.get(), nullptr, nullptr, nullptr));
  if (x509 == nullptr) {
    throw std::runtime_error("PEM_read_bio_X509: " + getErrors());
  }

  if (SSL_CTX_use_certificate(ctx_, x509.get()) == 0) {
    throw std::runtime_error("SSL_CTX_use_certificate: " + getErrors());
  }

  // Any further X509 PEM blocks are treated as additional certificates in
  // the certificate chain.
  constexpr size_t kMaxCertChain = 64;

  for (size_t i = 0; i < kMaxCertChain; i++) {
    x509.reset(PEM_read_bio_X509(bio.get(), nullptr, nullptr, nullptr));
    if (x509 == nullptr) {
      ERR_clear_error();
      return;
    }

    if (SSL_CTX_add1_chain_cert(ctx_, x509.get()) == 0) {
      throw std::runtime_error("SSL_CTX_add0_chain_cert: " + getErrors());
    }
  }

  throw std::runtime_error(
      "loadCertificateFromBufferPEM(): Too many certificates in chain");
}

void SSLContext::loadPrivateKey(const char* path, const char* format) {
  if (path == nullptr || format == nullptr) {
    throw std::invalid_argument(
        "loadPrivateKey: either <path> or <format> is nullptr");
  }
  if (strcmp(format, "PEM") == 0) {
    if (SSL_CTX_use_PrivateKey_file(ctx_, path, SSL_FILETYPE_PEM) == 0) {
      throw std::runtime_error("SSL_CTX_use_PrivateKey_file: " + getErrors());
    }
  } else {
    throw std::runtime_error(
        "Unsupported private key format: " + std::string(format));
  }
}

void SSLContext::loadPrivateKeyFromBufferPEM(folly::StringPiece pkey) {
  if (pkey.data() == nullptr) {
    throw std::invalid_argument("loadPrivateKey: <pkey> is nullptr");
  }

  ssl::BioUniquePtr bio(BIO_new(BIO_s_mem()));
  if (bio == nullptr) {
    throw std::runtime_error("BIO_new: " + getErrors());
  }

  int written = BIO_write(bio.get(), pkey.data(), int(pkey.size()));
  if (written <= 0 || static_cast<unsigned>(written) != pkey.size()) {
    throw std::runtime_error("BIO_write: " + getErrors());
  }

  ssl::EvpPkeyUniquePtr key(
      PEM_read_bio_PrivateKey(bio.get(), nullptr, nullptr, nullptr));
  if (key == nullptr) {
    throw std::runtime_error("PEM_read_bio_PrivateKey: " + getErrors());
  }

  if (SSL_CTX_use_PrivateKey(ctx_, key.get()) == 0) {
    throw std::runtime_error("SSL_CTX_use_PrivateKey: " + getErrors());
  }
}

void SSLContext::loadCertKeyPairFromBufferPEM(
    folly::StringPiece cert, folly::StringPiece pkey) {
  loadCertificateFromBufferPEM(cert);
  loadPrivateKeyFromBufferPEM(pkey);
  if (!isCertKeyPairValid()) {
    throw std::runtime_error("SSL certificate and private key do not match");
  }
}

void SSLContext::loadCertKeyPairFromFiles(
    const char* certPath,
    const char* keyPath,
    const char* certFormat,
    const char* keyFormat) {
  loadCertificate(certPath, certFormat);
  loadPrivateKey(keyPath, keyFormat);
  if (!isCertKeyPairValid()) {
    throw std::runtime_error("SSL certificate and private key do not match");
  }
}

void SSLContext::setCertChainKeyPair(
    std::vector<ssl::X509UniquePtr>&& certChain, ssl::EvpPkeyUniquePtr&& key) {
  if (certChain.empty()) {
    throw std::invalid_argument("Empty certificate chain provided");
  }

  constexpr size_t kMaxCertChain = 65;
  if (kMaxCertChain < certChain.size()) {
    throw std::invalid_argument("Too many certificates in chain");
  }

  if (SSL_CTX_use_PrivateKey(ctx_, key.get()) == 0) {
    throw std::runtime_error("SSL_CTX_use_PrivateKey: " + getErrors());
  }

  auto& leafCert = certChain.front();

  if (SSL_CTX_use_certificate(ctx_, leafCert.get()) == 0) {
    throw std::runtime_error("SSL_CTX_use_certificate: " + getErrors());
  }

  for (size_t i = 1; i < certChain.size(); ++i) {
    if (SSL_CTX_add1_chain_cert(ctx_, certChain[i].get()) == 0) {
      throw std::runtime_error("SSL_CTX_add0_chain_cert: " + getErrors());
    }
  }

  if (!isCertKeyPairValid()) {
    throw std::runtime_error("SSL certificate and private key do not match");
  }
}

bool SSLContext::isCertKeyPairValid() const {
  return SSL_CTX_check_private_key(ctx_) == 1;
}

void SSLContext::loadTrustedCertificates(const char* path) {
  if (path == nullptr) {
    throw std::invalid_argument("loadTrustedCertificates: <path> is nullptr");
  }
  if (SSL_CTX_load_verify_locations(ctx_, path, nullptr) == 0) {
    throw std::runtime_error("SSL_CTX_load_verify_locations: " + getErrors());
  }
  ERR_clear_error();
}

void SSLContext::loadTrustedCertificates(X509_STORE* store) {
  SSL_CTX_set_cert_store(ctx_, store);
}

void SSLContext::setSupportedClientCertificateAuthorityNames(
    std::vector<ssl::X509NameUniquePtr> names) {
  // SSL_CTX_set_client_CA_list *takes ownership* of a STACK_OF(X509_NAME)
  // where each pointer in the STACK is an *owned* X509.
  ssl::OwningStackOfX509NameUniquePtr nameList(sk_X509_NAME_new(nullptr));
  if (!nameList) {
    throw std::runtime_error(
        "SSLContext::setSupportedClientCertificateAuthorityNames failed to allocate name list");
  }

  // Release ownership of the X509_NAME into the stack.
  //
  // If exceptions are thrown, all elements are properly cleaned up because of
  // the *UniquePtr wrappers.
  for (auto& name : names) {
    if (!sk_X509_NAME_push(nameList.get(), name.release())) {
      throw std::runtime_error(
          "SSLContext::setSupportedClientCertificateAuthorityNames failed to add X509_NAME");
    }
  }

  // And finally pass ownership over the whole X509_NAME stack to OpenSSL
  SSL_CTX_set_client_CA_list(ctx_, nameList.release());
}

void SSLContext::passwordCollector(
    std::shared_ptr<ssl::PasswordCollector> collector) {
  if (collector == nullptr) {
    LOG(ERROR) << "passwordCollector: ignore invalid password collector";
    return;
  }
  collector_ = collector;
  SSL_CTX_set_default_passwd_cb(ctx_, passwordCallback);
  SSL_CTX_set_default_passwd_cb_userdata(ctx_, this);
}

void SSLContext::setServerNameCallback(const ServerNameCallback& cb) {
  serverNameCb_ = cb;
}

void SSLContext::addClientHelloCallback(const ClientHelloCallback& cb) {
  clientHelloCbs_.push_back(cb);
}

int SSLContext::baseServerNameOpenSSLCallback(SSL* ssl, int* al, void* data) {
  auto context = (SSLContext*)data;

  if (context == nullptr) {
    return SSL_TLSEXT_ERR_NOACK;
  }

  for (auto& cb : context->clientHelloCbs_) {
    // Generic callbacks to happen after we receive the Client Hello.
    // For example, we use one to switch which cipher we use depending
    // on the user's TLS version.  Because the primary purpose of
    // baseServerNameOpenSSLCallback is for SNI support, and these callbacks
    // are side-uses, we ignore any possible failures other than just logging
    // them.
    cb(ssl);
  }

  if (!context->serverNameCb_) {
    return SSL_TLSEXT_ERR_NOACK;
  }

  ServerNameCallbackResult ret = context->serverNameCb_(ssl);
  switch (ret) {
    case SERVER_NAME_FOUND:
      return SSL_TLSEXT_ERR_OK;
    case SERVER_NAME_NOT_FOUND:
      return SSL_TLSEXT_ERR_NOACK;
    case SERVER_NAME_NOT_FOUND_ALERT_FATAL:
      *al = TLS1_AD_UNRECOGNIZED_NAME;
      return SSL_TLSEXT_ERR_ALERT_FATAL;
    default:
      CHECK(false);
  }

  return SSL_TLSEXT_ERR_NOACK;
}

int SSLContext::alpnSelectCallback(
    SSL* /* ssl */,
    const unsigned char** out,
    unsigned char* outlen,
    const unsigned char* in,
    unsigned int inlen,
    void* data) {
  auto context = (SSLContext*)data;
  CHECK(context);
  if (context->advertisedNextProtocols_.empty()) {
    *out = nullptr;
    *outlen = 0;
  } else {
    auto i = context->pickNextProtocols();
    const auto& item = context->advertisedNextProtocols_[i];
    if (SSL_select_next_proto(
            (unsigned char**)out,
            outlen,
            item.protocols,
            item.length,
            in,
            inlen) != OPENSSL_NPN_NEGOTIATED) {
      if (!context->getAlpnAllowMismatch()) {
        return SSL_TLSEXT_ERR_ALERT_FATAL;
      } else {
        return SSL_TLSEXT_ERR_NOACK;
      }
    }
  }
  return SSL_TLSEXT_ERR_OK;
}

std::string SSLContext::getAdvertisedNextProtocols() const {
  if (advertisedNextProtocols_.empty()) {
    return "";
  }
  std::string alpns(
      (const char*)advertisedNextProtocols_[0].protocols + 1,
      advertisedNextProtocols_[0].length - 1);
  auto len = advertisedNextProtocols_[0].protocols[0];
  for (size_t i = len; i < alpns.length();) {
    len = alpns[i];
    alpns[i] = ',';
    i += len + 1;
  }
  return alpns;
}

bool SSLContext::setAdvertisedNextProtocols(
    const std::list<std::string>& protocols) {
  return setRandomizedAdvertisedNextProtocols({{1, protocols}});
}

bool SSLContext::setRandomizedAdvertisedNextProtocols(
    const std::list<NextProtocolsItem>& items) {
  unsetNextProtocols();
  if (items.empty()) {
    return false;
  }
  int total_weight = 0;
  for (const auto& item : items) {
    if (item.protocols.empty()) {
      continue;
    }
    AdvertisedNextProtocolsItem advertised_item;
    advertised_item.length = 0;
    for (const auto& proto : item.protocols) {
      ++advertised_item.length;
      auto protoLength = proto.length();
      if (protoLength >= 256) {
        deleteNextProtocolsStrings();
        return false;
      }
      advertised_item.length += unsigned(protoLength);
    }
    advertised_item.protocols = new unsigned char[advertised_item.length];
    if (!advertised_item.protocols) {
      throw std::runtime_error("alloc failure");
    }
    unsigned char* dst = advertised_item.protocols;
    for (auto& proto : item.protocols) {
      auto protoLength = uint8_t(proto.length());
      *dst++ = (unsigned char)protoLength;
      memcpy(dst, proto.data(), protoLength);
      dst += protoLength;
    }
    total_weight += item.weight;
    advertisedNextProtocols_.push_back(advertised_item);
    advertisedNextProtocolWeights_.push_back(item.weight);
  }
  if (total_weight == 0) {
    deleteNextProtocolsStrings();
    return false;
  }
  nextProtocolDistribution_ = std::discrete_distribution<>(
      advertisedNextProtocolWeights_.begin(),
      advertisedNextProtocolWeights_.end());
  SSL_CTX_set_alpn_select_cb(ctx_, alpnSelectCallback, this);
  // Client cannot really use randomized alpn
  // Note that this function reverses the typical return value convention
  // of openssl and returns 0 on success.
  return SSL_CTX_set_alpn_protos(
             ctx_,
             advertisedNextProtocols_[0].protocols,
             advertisedNextProtocols_[0].length) == 0;
}

void SSLContext::deleteNextProtocolsStrings() {
  for (auto protocols : advertisedNextProtocols_) {
    delete[] protocols.protocols;
  }
  advertisedNextProtocols_.clear();
  advertisedNextProtocolWeights_.clear();
}

void SSLContext::unsetNextProtocols() {
  deleteNextProtocolsStrings();
  SSL_CTX_set_alpn_select_cb(ctx_, nullptr, nullptr);
  SSL_CTX_set_alpn_protos(ctx_, nullptr, 0);
  // clear the error stack here since openssl internals sometimes add a
  // malloc failure when doing a memdup of NULL, 0..
  ERR_clear_error();
}

size_t SSLContext::pickNextProtocols() {
  CHECK(!advertisedNextProtocols_.empty()) << "Failed to pickNextProtocols";
  auto rng = ThreadLocalPRNG();
  return size_t(nextProtocolDistribution_(rng));
}

SSL* SSLContext::createSSL() const {
  SSL* ssl = SSL_new(ctx_);
  if (ssl == nullptr) {
    throw std::runtime_error("SSL_new: " + getErrors());
  }
  return ssl;
}

void SSLContext::setSessionCacheContext(const std::string& context) {
  SSL_CTX_set_session_id_context(
      ctx_,
      reinterpret_cast<const unsigned char*>(context.data()),
      std::min<unsigned int>(
          static_cast<unsigned int>(context.length()), SSL_MAX_SID_CTX_LENGTH));
}

/**
 * Match a name with a pattern. The pattern may include wildcard. A single
 * wildcard "*" can match up to one component in the domain name.
 *
 * @param  host    Host name, typically the name of the remote host
 * @param  pattern Name retrieved from certificate
 * @param  size    Size of "pattern"
 * @return True, if "host" matches "pattern". False otherwise.
 */
bool SSLContext::matchName(const char* host, const char* pattern, int size) {
  bool match = false;
  int i = 0, j = 0;
  while (i < size && host[j] != '\0') {
    if (toupper(pattern[i]) == toupper(host[j])) {
      i++;
      j++;
      continue;
    }
    if (pattern[i] == '*') {
      while (host[j] != '.' && host[j] != '\0') {
        j++;
      }
      i++;
      continue;
    }
    break;
  }
  if (i == size && host[j] == '\0') {
    match = true;
  }
  return match;
}

int SSLContext::passwordCallback(char* password, int size, int, void* data) {
  auto context = (SSLContext*)data;
  if (context == nullptr || context->passwordCollector() == nullptr) {
    return 0;
  }
  std::string userPassword;
  // call user defined password collector to get password
  context->passwordCollector()->getPassword(userPassword, size);
  auto const length = std::min(userPassword.size(), size_t(size));
  std::memcpy(password, userPassword.data(), length);
  return int(length);
}

#if defined(SSL_MODE_HANDSHAKE_CUTTHROUGH)
void SSLContext::enableFalseStart() {
  SSL_CTX_set_mode(ctx_, SSL_MODE_HANDSHAKE_CUTTHROUGH);
}
#endif

void SSLContext::setOptions(long options) {
  long newOpt = SSL_CTX_set_options(ctx_, options);
  if ((newOpt & options) != options) {
    throw std::runtime_error("SSL_CTX_set_options failed");
  }
}

std::string SSLContext::getErrors(int errnoCopy) {
  std::string errors;
  unsigned long errorCode;
  char message[256];

  errors.reserve(512);
  while ((errorCode = ERR_get_error()) != 0) {
    if (!errors.empty()) {
      errors += "; ";
    }
    const char* reason = ERR_reason_error_string(errorCode);
    if (reason == nullptr) {
      snprintf(message, sizeof(message) - 1, "SSL error # %08lX", errorCode);
      reason = message;
    }
    errors += reason;
  }
  if (errors.empty()) {
    errors = "error code: " + folly::to<std::string>(errnoCopy);
  }
  return errors;
}

void SSLContext::disableTLS13() {
  SSL_CTX_set_max_proto_version(ctx_, TLS1_2_VERSION);
}

void SSLContext::setupCtx(SSL_CTX* ctx) {
  // 1) folly::AsyncSSLSocket wants to unconditionally store a client
  // session, so that is possible to later perform TLS resumption.
  // For that, we need SSL_SESS_CACHE_CLIENT.
  //
  // 2) wangle::SSLSessionCacheManager needs to be able to receive
  // SSL_SESSIONs that are established through a successful
  // connection. For that, we need SSL_SESS_CACHE_SERVER. Consequently,
  // given the requirements of (1), we opt to use SSL_SESS_CACHE_BOTH
  //
  // 3) We explicitly disable the OpenSSL internal session cache, as there
  // is very little we can do to control the memory usage of the internal
  // session cache. Server side session-id based caching should be explicitly
  // opted-in by the user, by forcing them to provide an implementation of
  // a SessionCache interface (e.g. wangle::SSLSessionCacheManager); i.e.,
  // the user must be cognizant of the fact that doing so would result in
  // increased memory usage.
  SSL_CTX_set_session_cache_mode(
      ctx,
      SSL_SESS_CACHE_BOTH | SSL_SESS_CACHE_NO_INTERNAL |
          SSL_SESS_CACHE_NO_AUTO_CLEAR);

  SSL_CTX_set_ex_data(ctx, getExDataIndex(), this);
  SSL_CTX_sess_set_new_cb(ctx, SSLContext::newSessionCallback);
}

SSLContext* SSLContext::getFromSSLCtx(const SSL_CTX* ctx) {
  return static_cast<SSLContext*>(SSL_CTX_get_ex_data(ctx, getExDataIndex()));
}

int SSLContext::newSessionCallback(SSL* ssl, SSL_SESSION* session) {
  SSL_CTX* ctx = SSL_get_SSL_CTX(ssl);
  SSLContext* context = getFromSSLCtx(ctx);

  auto& cb = context->sessionLifecycleCallbacks_;
  if (cb != nullptr && cb) {
    SSL_SESSION_up_ref(session);
    auto sessionPtr = folly::ssl::SSLSessionUniquePtr(session);
    cb->onNewSession(ssl, std::move(sessionPtr));
  }

  // Session will either be moved to session manager or
  // freed when the unique_ptr goes out of scope
  auto sessionPtr = folly::ssl::SSLSessionUniquePtr(session);
  auto sessionManager = folly::ssl::SSLSessionManager::getFromSSL(ssl);
  if (sessionManager) {
    sessionManager->onNewSession(std::move(sessionPtr));
  }

  return 1;
}

void SSLContext::setSessionLifecycleCallbacks(
    std::unique_ptr<SessionLifecycleCallbacks> cb) {
  sessionLifecycleCallbacks_ = std::move(cb);
}

void SSLContext::setCiphersuitesOrThrow(const std::string& ciphersuites) {
  auto rc = SSL_CTX_set_ciphersuites(ctx_, ciphersuites.c_str());
  if (rc == 0) {
    throw std::runtime_error("SSL_CTX_set_ciphersuites: " + getErrors());
  }
}

void SSLContext::setAllowNoDheKex(bool flag) {
  auto opt = SSL_OP_ALLOW_NO_DHE_KEX;
  if (flag) {
    SSL_CTX_set_options(ctx_, opt);
  } else {
    SSL_CTX_clear_options(ctx_, opt);
  }
}

void SSLContext::setTicketHandler(
    std::unique_ptr<OpenSSLTicketHandler> handler) {
  ticketHandler_ = std::move(handler);
  SSL_CTX_set_tlsext_ticket_key_cb(ctx_, dispatchTicketCrypto);
}

} // namespace folly
