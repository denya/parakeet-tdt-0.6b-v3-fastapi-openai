"""Preserve the deployed eMudhra/AAA distrust policy in every Python trust store."""
import pathlib
import re
import ssl
import sys

BLOCKED = (b'emSign ECC Root CA - G3', b'AAA Certificate Services')

def filtered(text):
    def replace(match):
        der = ssl.PEM_cert_to_DER_cert(match.group())
        return '' if any(name in der for name in BLOCKED) else match.group()
    return re.sub(r'-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----', replace, text, flags=re.S)

if __name__ == '__main__':
    if '--self-test' in sys.argv:
        good = ssl.DER_cert_to_PEM_cert(b'kept certificate')
        for name in BLOCKED:
            assert filtered(ssl.DER_cert_to_PEM_cert(name) + good).strip() == good.strip()
        assert filtered('unrelated') == 'unrelated'
    else:
        bundles = [pathlib.Path('/etc/ssl/certs/ca-certificates.crt')]
        bundles += list(pathlib.Path('/usr/local/lib').glob('python*/site-packages/**/certifi/cacert.pem'))
        for bundle in bundles:
            text = filtered(bundle.read_text())
            assert text.count('BEGIN CERTIFICATE') > 100, bundle
            bundle.write_text(text)
        for directory in ['/usr/share/ca-certificates', '/etc/ssl/certs']:
            for cert in pathlib.Path(directory).rglob('*'):
                if cert.is_file() and cert.suffix in {'.pem', '.crt'}:
                    text = cert.read_text(errors='replace')
                    if 'BEGIN CERTIFICATE' in text and not filtered(text).strip():
                        cert.unlink()
