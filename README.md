(WIP)

Comparing images and giving a similarity score in percentage using AI.

---

Install dependencies:

```sh
pip install transformers torch pillow
```

You need a base image in the root folder called `base.jpg` (or just change it in the script).

And then you just pass a list of other images to `compare_to_other_images([image list here])`.

Run with `python compare.py`, duh.

---

On Apple Silicone chips (ARM, i.e. M1...), it only runs with python3.12 (or earlier, but not sure which),
because of (something related to) the `transformers` dependency. `pip install transformers` doesn't work
and you have to do some gymnastics.
