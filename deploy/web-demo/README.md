# Publish the static Iris demo

The hosting service serves the tested artifact produced by
[`examples/web-demo/export.py`](../../examples/web-demo/README.md). Publication
copies those exact bytes; it does not rebuild the site on the server or start
Iris's Python API. Visitors' annotations remain in their browser.

Build the artifact with its full source commit recorded in the manifest:

```sh
revision=$(git rev-parse HEAD)
python3 examples/web-demo/export.py --output dist/demo --revision "$revision"
```

## Host layout

Provision a dedicated deployment account with write access only to its site
directory. Install `receive.py` as a root-owned, non-writable executable outside
that directory, for example `/usr/local/lib/iris-demo/receive.py`. It uses Python
3's standard library on Linux, including `fcntl.flock`.

```text
/srv/iris-demo/
  nginx.conf                  # administrator-managed, read only in the container
  site/                       # writable by the restricted deployment account
    .deploy.lock
    assets/<asset-version>/   # immutable, shared across releases
    releases/<git-revision>/
      index.html
      manifest.json
    current -> releases/<git-revision>
    previous -> releases/<previous-git-revision>
```

Serve `site/current` as the document root and `/assets/` from `site/assets`.
Keeping these separate preserves assets used by pages opened before an update.
Do not mount `current` alone: a bind mount of its resolved target would not
follow later atomic symlink replacements. Directories must be readable and
traversable by the container's unprivileged user; published directories use
mode 0755 and files use 0644.

## Coolify service

Create a dedicated Compose application using
[`compose.coolify.yaml`](compose.coolify.yaml). Copy [`nginx.conf`](nginx.conf)
to the administrator-managed host path first. The Compose service mounts the
site and configuration read only, exposes container port 8080 without binding
host ports, and uses the existing Coolify proxy for its public HTTPS domain.
It needs no application environment variables, database or Coolify API token.

The runtime is limited to 0.5 CPU, 128 MiB memory and 128 processes. Its root
filesystem is read only; `/tmp` is a bounded writable tmpfs. All capabilities
are dropped and privilege escalation is disabled. The
[upstream unprivileged Nginx image](https://github.com/nginx/docker-nginx-unprivileged)
is pinned by its multi-platform OCI digest in Compose. Review and test a new
digest explicitly when updating the web server.

`/healthz` returns the active index only if it exists and is readable; before
the first release the service is unhealthy. After deployment, check the public
index, manifest, referenced assets and expected revision as well as health.
`/api` and `/api/…`, hidden paths and missing files return 404. HTML and the
manifest require cache revalidation; versioned assets carry
`Cache-Control: public, max-age=31536000, immutable`.

## Restricted SSH transport

Keep the deployment account outside privileged groups, without sudo or Docker
access. Disable password authentication and install one dedicated public key
with a forced command and OpenSSH restrictions. For example, its root-managed
`authorized_keys` entry begins with:

```text
restrict,command="/usr/bin/python3 /usr/local/lib/iris-demo/receive.py --root /srv/iris-demo/site --ssh" ssh-ed25519 <deployment-public-key>
```

The key cannot provide a shell, SFTP, port forwarding, agent forwarding or a
PTY. The account must not be able to edit its own SSH restrictions or the
receiver. Pin the host's verified SSH host key in the publication environment;
do not learn and trust a new key during each deployment.

The forced command accepts exactly one of these original SSH commands:

```text
deploy <40-character-lowercase-Git-SHA>
rollback <expected-current-40-character-lowercase-Git-SHA>
```

A deploy reads a ZIP from standard input. Put `index.html`, `manifest.json` and
the `assets/` files directly at the archive root, without an enclosing `demo/`
directory. The manifest must have format `iris-web-demo-site-v1` and its
`source_revision` must equal the command's Git SHA. Its complete `files` mapping
must match the archive's file paths, sizes and SHA-256 hashes. Structural
directory entries are allowed; extra files, extra directories, duplicate paths,
symlinks, special files, traversal paths and encrypted ZIPs are rejected.

Limits are 32 MiB for the compressed input, 128 MiB uncompressed, 4096 ZIP
entries and 4 MiB for the manifest. Only stored or deflated ZIP entries are
accepted. The receiver never executes uploaded files or extracts archive paths
with a general-purpose ZIP extraction command.

Every mutation of release state runs under one exclusive file lock. The receiver
validates the entire archive before publication, writes and verifies immutable
assets and release files, then atomically replaces `current`. A repeated deploy
of the same revision and bytes is idempotent. Reusing a revision or asset version
for different bytes is refused. Failed validation or transfer preserves the
current release. No old assets or releases are automatically deleted.

## Verification and rollback

Successful commands write one JSON object to standard output with `action`,
`revision`, `previous` and `asset_version`. `previous` is a Git SHA or `null`.
Failures exit nonzero and write a diagnostic to standard error. If SSH ends
before its acknowledgement arrives, inspect the public manifest and content
before deciding whether deployment succeeded; do not assume the old release is
still active.

After activating a new revision, the publication job must verify the expected
public manifest and every artifact byte. If verification fails, send
`rollback <failed-revision>`. The receiver restores the previous intact release
only when `current` still equals that expected revision. It refuses to roll back
over a newer deployment. The first release has no automatic rollback target.
Rollback swaps the two pointers and does not remove any immutable files.

An administrator can use the same protocol locally:

```sh
python3 receive.py --root /srv/iris-demo/site deploy "$revision" < artifact.zip
python3 receive.py --root /srv/iris-demo/site rollback "$revision"
```

## Enable automatic publication

The publication job is gated on successful checks for the source revision and
the repository variable `IRIS_DEMO_DEPLOY_ENABLED=true`. It uses the protected
GitHub environment `iris-demo`; add environment approval rules if required.
Enable the variable only after the receiver, forced SSH command, host-key pin,
initial artifact and public HTTPS checks have been verified.

Configure these GitHub Actions secrets in that environment or repository:

- `IRIS_DEMO_SSH_KEY`: the dedicated restricted deployment private key.
- `IRIS_DEMO_KNOWN_HOSTS`: the verified SSH host-key entry for the connection's
  `HostKeyAlias`, `iris-demo-vps`.

Optional repository variables are `IRIS_DEMO_HOST` (default
`iris.victormonnot.com`) and `IRIS_DEMO_USER` (default `iris-demo`). No hosting
admin password, Coolify API credential or unrelated application's secret is
needed. Keep secret values outside the repository and workflow logs.

The publication client verifies the manifest and every public artifact byte,
including when the SSH acknowledgement is lost. A failed verification requests
the guarded rollback described above. Repository code changes take effect only
after the checked artifact has been published successfully.

Keep normal host backups and monitor disk space. Retention or cleanup is a
separate administrative operation: account for currently published releases,
rollback targets and visitors still using older cached pages before removing
anything. Preserve the public artifact's software licences and sample credits.
