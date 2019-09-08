The part refiner is used to refine the initial generated shapes with res 32 from genertor to more realistic shape with res 64.

## Preparing the training data
In the folder of "data_preparing_for_PR", we need to prepare the training data for part refiner, and initial data for PR can be downloaded from https://drive.google.com/open?id=1nTEBVyKg2UEwmrlewb9TLCc-ILGmq2_5

Step 1: On the folder named "seperate_training_data", we use the "seperate_part_mat.py" file to seperate each semantic parts from training data and combine these parts into a big npy file.

Step2: On the "gen_pair" folder, we need to make input and output pair for the part refiner. 

## Training the part refiner

```sh(take PR for chair as example)
python train_part_ae_v3_chair.py
```
Notice that, only epoch = 4000 is enough for training the part refiner


## Using the part refiner to refine the initial shape from res 32 to res 64
```sh(take PR for chair as example)
python ae_generate_results.py
```
Notice to change the category_4 for different categories.
