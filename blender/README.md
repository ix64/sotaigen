# sotaigen-blender

The Blender half of [sotaigen](https://github.com/ix64/sotaigen): the scripts that turn a body into a
rigged `.blend`, pose one, and swap its head for one made elsewhere. They import Blender's Python API,
so this package is [GPL-3.0-or-later](LICENSE); `sotaigen` itself is Apache-2.0 and runs these scripts
inside a Blender process of their own, without importing them.

Install it next to `sotaigen` to write `.blend` files:

```sh
pip install sotaigen sotaigen-blender
sotaigen blend body.json body.blend
```

It needs Blender 5.2 or later, found through the `BLENDER` environment variable, then `blender` on
`PATH`, then the macOS app. The two packages are released together with the same version, and each
refuses the other's data when their versions disagree.
