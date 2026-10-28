# Build environment notes (no-root machines)

`ort`'s "download-binaries" build path pulls ONNX Runtime over HTTPS using a
client that links OpenSSL, which needs `pkg-config` + headers at build time.
On a machine where you cannot `apt-get install pkg-config libssl-dev`, the
following user-local workaround was used successfully (Debian trixie):

```sh
mkdir -p ~/opt/voicer-deps && cd ~/opt/voicer-deps
apt-get download libssl-dev zlib1g-dev libzstd-dev pkgconf   # dpkg -x each into ./
# pkgconf's real binary may be missing; a shim works (see below)
# headers: opensslconf.h/configuration.h live in usr/include/x86_64-linux-gnu/openssl
#   -> symlink both into usr/include/openssl/ (or a merged dir)
# runtime libs: symlink /usr/lib/x86_64-linux-gnu/lib{ssl,crypto}.so.3 next to the .so links
```

A minimal `pkg-config` shim (`~/opt/voicer-deps/usr/bin/pkgconf`) that answers
openssl queries with `-I<merged-include>` / `-L<libdir> -lssl -lcrypto -lzstd
-lz -ldl -pthread` is enough for openssl-sys.

Other packages extracted as needed: `libclang1-19`, `libclang-common-19-dev`
(bindgen; set `LIBCLANG_PATH` and symlink the clang resource dir),
`cmake`, `libllvm19`, `libjsoncpp26`, `librhash1`, `libuv1` (espeak-rs-sys
builds espeak-ng with cmake via piper-rs).

Then build with:

```sh
export PATH=~/opt/voicer-deps/usr/bin:$PATH
export LIBCLANG_PATH=~/opt/voicer-deps/usr/lib/x86_64-linux-gnu
export BINDGEN_EXTRA_CLANG_ARGS=-I$HOME/opt/voicer-deps/usr/lib/llvm-19/lib/clang/19/include
export RUSTFLAGS="-L $HOME/opt/voicer-deps/usr/lib/x86_64-linux-gnu"
cargo build --release
```

If you have root, just: `apt-get install pkg-config libssl-dev clang cmake`.
