# recog

pix2pix, tensorflow, and a webcam. Complete hack.

```
pip3 install -r requirements.txt
```

```
# because we use the browser's webcam, we must serve over SSL locally.
# generate self-signed cert/key like this:
openssl req -newkey rsa:2048 -new -nodes -x509 -days 3650 -keyout key.pem -out cert.pem

# ...then run a local server (e.g.):
http-server --ssl --cert cert.pem

# ...then go to https://0.0.0.0:8080/index.html
```
