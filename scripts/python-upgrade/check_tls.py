"""Strict TLS controls with disposable local certificates, never system trust edits."""
import argparse
from pathlib import Path
import socket
import ssl
import subprocess
import tempfile
import threading
import urllib.request

import certifi


def command(*args):
    result = subprocess.run(['openssl', *map(str, args)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr)


def handshake(cert, key, context, hostname='localhost', expected_error=None):
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(cert, key)
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        listener.settimeout(10)
        def serve():
            connection, _ = listener.accept()
            try:
                with server_context.wrap_socket(connection, server_side=True):
                    pass
            except ssl.SSLError:
                connection.close()  # Expected when the client rejects the certificate.
        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        try:
            with socket.create_connection(listener.getsockname(), timeout=10) as connection:
                try:
                    with context.wrap_socket(connection, server_hostname=hostname):
                        if expected_error is not None:
                            raise AssertionError('Invalid certificate accepted')
                except ssl.SSLCertVerificationError as error:
                    if expected_error is None or error.verify_code not in expected_error:
                        raise
        finally:
            thread.join(timeout=10)
            assert not thread.is_alive(), 'TLS test server did not finish'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--public', action='store_true', help='Also require public HTTPS with each trust store')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='parakeet-tls-') as directory:
        root = Path(directory)
        ca, ca_key = root/'ca.pem', root/'ca.key'
        key, csr = root/'leaf.key', root/'leaf.csr'
        command('req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '1',
                '-subj', '/CN=Disposable Parakeet TLS test CA', '-keyout', ca_key, '-out', ca,
                '-addext', 'basicConstraints=critical,CA:TRUE',
                '-addext', 'keyUsage=critical,keyCertSign,cRLSign')
        command('req', '-new', '-newkey', 'rsa:2048', '-nodes', '-subj', '/CN=localhost',
                '-keyout', key, '-out', csr)
        extensions = root/'extensions.cnf'
        extensions.write_text('subjectAltName=DNS:localhost\nbasicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\n')
        command('x509', '-req', '-in', csr, '-CA', ca, '-CAkey', ca_key,
                '-set_serial', '1', '-days', '1', '-extfile', extensions, '-out', root/'valid.pem')
        # Explicit past dates work across OpenSSL 3.0/3.5; negative -days does not.
        (root/'index').write_text('')
        (root/'serial').write_text('02\n')
        config = root/'ca.cnf'
        config.write_text(f"[ca]\ndefault_ca=test\n[test]\ndatabase={root}/index\nserial={root}/serial\nnew_certs_dir={root}\ncertificate={ca}\nprivate_key={ca_key}\ndefault_md=sha256\npolicy=policy\nx509_extensions=server\n[policy]\ncommonName=supplied\n[server]\n" + extensions.read_text())
        command('ca', '-batch', '-config', config, '-in', csr, '-out', root/'expired.pem',
                '-startdate', '200101000000Z', '-enddate', '210101000000Z')
        for name, cafile in [('system', None), ('certifi', certifi.where())]:
            context = ssl.create_default_context(cafile=cafile)
            assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
            if args.public:
                with urllib.request.urlopen('https://pypi.org', context=context, timeout=20) as response:
                    assert response.status == 200
            handshake(root/'valid.pem', key, context, expected_error={20, 21})
            # Trust the generated CA only in this disposable test context. No files/settings modified.
            context.load_verify_locations(cafile=ca)
            handshake(root/'valid.pem', key, context)
            handshake(root/'expired.pem', key, context, expected_error={10})
            handshake(root/'valid.pem', key, context, hostname='wrong.invalid', expected_error={62})
            print(name+': valid control accepted; untrusted, expired and wrong-host certificates rejected')


if __name__ == '__main__':
    main()
