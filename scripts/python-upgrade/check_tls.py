"""Verify actual certificate validation; run outside a TLS-intercepting proxy."""
import ssl
import urllib.error
import urllib.request
import certifi

for name,ctx in [('system',ssl.create_default_context()),('certifi',ssl.create_default_context(cafile=certifi.where()))]:
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname
    with urllib.request.urlopen('https://pypi.org',context=ctx,timeout=20) as r:
        assert r.status==200
    for url in ['https://expired.badssl.com','https://self-signed.badssl.com','https://wrong.host.badssl.com']:
        try:urllib.request.urlopen(url,context=ctx,timeout=20)
        except urllib.error.URLError as exc:
            if not isinstance(exc.reason,ssl.SSLCertVerificationError):raise
        else:raise AssertionError(name+' accepted invalid certificate: '+url)
    print(name+': valid control accepted; expired, self-signed and wrong-host certificates rejected')
