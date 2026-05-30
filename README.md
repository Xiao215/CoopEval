# CoopEval Website

This directory contains the static project website published by GitHub Pages.

The site is deployed by `.github/workflows/pages.yml` using GitHub Actions. The
workflow uploads this `web/` directory as the Pages artifact, so the repository
does not need to use GitHub's special `docs/` publishing mode.

After enabling GitHub Pages with **Source: GitHub Actions**, pushes to `main`
that touch `web/**` will publish the site at:

```text
https://<username>.github.io/<repository-name>/
```

For the public repository, that resolves to:

```text
https://xiao215.github.io/CoopEval/
```

Keep asset paths relative, for example `static/images/...`, so the site works
under the repository subpath.
