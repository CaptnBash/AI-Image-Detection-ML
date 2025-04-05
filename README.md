# AI Image detection

## utils
 
[download-laion-images.py](util/download-laion-images.py)

Downloads a specified amount of images (default 200'000) from [huggingface](https://huggingface.co/datasets/laion/laion-high-resolution).
###
```
folder = "laion-images" # change to whatever folder
login(token="your-token") # input your huggingface token

images_count = 200000
```
Change these lines and start the script.

The token can be generated [here](https://huggingface.co/settings/tokens).

## source for real images:

https://huggingface.co/datasets/laion/laion-high-resolution 

## source for AI and real images:

https://www.kaggle.com/datasets/superpotato9/dalle-recognition-dataset  (only DALL-E)

