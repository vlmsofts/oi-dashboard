"""
send_oi_whatsapp.py
Upload the 4 OI Monitor WhatsApp PNGs (CT, KC, CC, SB) to R2, then send via Twilio.
Order: Cotton -> Coffee -> Cocoa -> Sugar (matches build_whatsapp_oi.py's commodity loop).

Modeled on cot dashboard/send_cot_whatsapp.py -- same R2 bucket, same Twilio call shape.
"""
import base64, pathlib, time, urllib.parse, urllib.request, urllib.error
import boto3
from botocore.config import Config

# Credentials come from the gitignored .env (or real env vars), never from this
# tracked file -- see vlm_secrets.py. require() fails loudly at import if a value
# is missing, which is far better than a confusing 401 from Twilio mid-send.
import vlm_secrets

# WhatsApp/Twilio does NOT guarantee delivery order for messages submitted
# back-to-back -- without pacing, the 4 commodity images can arrive out of order.
# A short gap between sends keeps them in order (same fix used by the COT sender).
SEND_DELAY_SECS = 2.0

# ── Config ────────────────────────────────────────────────────────────────────
# Secrets resolve from env var -> .env (see vlm_secrets). R2_BUCKET and
# R2_PUBLIC_BASE stay inline: they are not secrets (the bucket is public-read)
# and keeping them here means the destination is visible in review. Same bucket
# the COT/export-sales/drought/cotton-on-call senders already use; OI gets its
# own 'oi/' key prefix so uploads never collide with theirs.
CF_ACCOUNT_ID    = vlm_secrets.require('CF_ACCOUNT_ID')
R2_ACCESS_KEY_ID = vlm_secrets.require('R2_ACCESS_KEY_ID')
R2_SECRET_KEY    = vlm_secrets.require('R2_SECRET_KEY')
R2_BUCKET        = 'crop-media'
R2_PUBLIC_BASE   = 'https://pub-d2bcb18933764e9183b558b140baba19.r2.dev'

TWILIO_SID   = vlm_secrets.require('TWILIO_SID')
TWILIO_TOKEN = vlm_secrets.require('TWILIO_TOKEN')
FROM_WA      = vlm_secrets.require('FROM_WA')
TO_WA        = vlm_secrets.require('TO_WA')

COMMODITY_ORDER = [
    ('CT', 'COTTON'),
    ('KC', 'COFFEE'),
    ('CC', 'COCOA'),
    ('SB', 'SUGAR'),
]


# ── R2 Upload ─────────────────────────────────────────────────────────────────
def upload_to_r2(png_paths, as_of):
    s3 = boto3.client(
        's3',
        endpoint_url          = f'https://{CF_ACCOUNT_ID}.r2.cloudflarestorage.com',
        aws_access_key_id     = R2_ACCESS_KEY_ID,
        aws_secret_access_key = R2_SECRET_KEY,
        config                = Config(signature_version='s3v4'),
        region_name           = 'auto',
    )
    public_urls = []
    for path in png_paths:
        key = f'oi/{as_of}/{pathlib.Path(path).name}'
        s3.upload_file(
            str(path), R2_BUCKET, key,
            ExtraArgs={'ContentType': 'image/png'},
        )
        url = f'{R2_PUBLIC_BASE}/{key}'
        public_urls.append(url)
        print(f'  R2 uploaded: {key}')
    return public_urls


# ── WhatsApp Send ─────────────────────────────────────────────────────────────
def send_whatsapp_image(url, commodity_name, as_of):
    api_url = f'https://api.twilio.com/2010-04-01/Accounts/{TWILIO_SID}/Messages.json'
    auth    = (TWILIO_SID + ':' + TWILIO_TOKEN).encode('utf-8')
    headers = {
        'Authorization': 'Basic ' + base64.b64encode(auth).decode('utf-8'),
        'Content-Type':  'application/x-www-form-urlencoded',
    }
    data = urllib.parse.urlencode({
        'From':     FROM_WA,
        'To':       TO_WA,
        'Body':     f'VLM Open Interest Monitor — {commodity_name} — {as_of}',
        'MediaUrl': url,
    }).encode('utf-8')
    req = urllib.request.Request(api_url, data=data, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            print(f'  WhatsApp sent [{commodity_name}]: HTTP {resp.status}')
            return True
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8', errors='replace')
        print(f'  WARNING: WhatsApp failed [{commodity_name}]: {e.code} {body[:300]}')
        return False
    except Exception as e:
        # URLError / timeout / DNS -- counted so the caller can alert instead of
        # the whole run dying mid-send.
        print(f'  WARNING: WhatsApp failed [{commodity_name}]: {e}')
        return False


# ── Main ──────────────────────────────────────────────────────────────────────
def main(whatsapp_dir=None):
    """
    whatsapp_dir: path to the dated output/whatsapp/<date>/ folder.
    If None, finds the most recent dated subfolder automatically.
    """
    base = pathlib.Path('output/whatsapp')
    if whatsapp_dir:
        dated_dir = pathlib.Path(whatsapp_dir)
    else:
        subdirs = sorted([d for d in base.iterdir() if d.is_dir()], reverse=True)
        if not subdirs:
            print('ERROR: No dated subfolders found in output/whatsapp/')
            return {'sent': 0, 'failed': 0, 'errors': ['no dated whatsapp subfolder found']}
        dated_dir = subdirs[0]

    as_of = dated_dir.name
    print(f'\nOI WhatsApp send — as of: {as_of}')
    print(f'Source folder: {dated_dir}\n')

    total_sent = 0
    total_failed = 0
    errors = []

    for i, (code, name) in enumerate(COMMODITY_ORDER, 1):
        png = dated_dir / f'OI_Monitor_{code}_{as_of}.png'
        if not png.exists():
            print(f'  [{code}] No PNG found ({png.name}) — skipping')
            continue

        print(f'[{code}] {name} — uploading...')
        try:
            urls = upload_to_r2([png], as_of)
        except Exception as e:
            # R2 upload has no retry; count it and move on so the other
            # commodities still attempt, then alert via the returned tally.
            print(f'  [{code}] R2 upload FAILED: {e}')
            total_failed += 1
            errors.append(f'{name}: R2 upload failed ({e})')
            continue

        print(f'[{code}] {name} — sending WhatsApp...')
        ok = send_whatsapp_image(urls[0], name, as_of)
        if ok:
            total_sent += 1
        else:
            total_failed += 1
            errors.append(f'{name}: WhatsApp send failed')

        # Pace every send (a trailing sleep after the last image is harmless) so
        # WhatsApp delivers the 4 commodity images in order.
        if i < len(COMMODITY_ORDER) and SEND_DELAY_SECS > 0:
            time.sleep(SEND_DELAY_SECS)

    summary = {'sent': total_sent, 'failed': total_failed, 'errors': errors}
    print(f'\nDone. WhatsApp sent={total_sent} failed={total_failed}')
    return summary


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=None, help='Path to dated whatsapp folder (optional)')
    args = ap.parse_args()
    main(whatsapp_dir=args.dir)
