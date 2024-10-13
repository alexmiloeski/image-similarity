## Description

Comparing images and giving a similarity score in percentage using AI.

This project uses OpenAI's CLIP deep learning model to get semantic information out of images and compare their similarity.

Workflow outline:

- load the images into memory
- create embeddings for them (convert them to vectors)
- run a cosine similarity function on the embeddings to get a value between -1 and 1
- convert that number to a percentage value

---

## Installing and running

Install dependencies:

```sh
pip install transformers torch pillow
```

You need a base image in the root folder called `base.jpg` (or just change it in the script).

And then you just pass a list of other images to `compare_to_other_images([image list here])`.

Run with `python compare.py`, duh.

---

### Note about Apple ARM chips

On Apple Silicone chips (ARM, i.e. M1...), it only runs with python3.12 (or earlier, but not sure which),
because of (something related to) the `transformers` dependency. `pip install transformers` doesn't work
and you have to do some gymnastics.
