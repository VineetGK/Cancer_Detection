# Breast Ultrasound Lesion Segmentation (Attention U-Net)

`Cancer_Detection.ipynb` trains an attention U-Net to **segment** breast lesions in ultrasound images, labelling every pixel as background, benign or malignant (3 classes), and scores it with the Dice coefficient.
It is an image segmentation model, not an image classifier, and it does not use an LLM.

This repo also contains `Final_Project.ipynb`, an unrelated LSTM trading-strategy notebook (see the section at the end).

## Data

- Breast ultrasound images and masks in the BUSI format (`benign/`, `malignant/`, `normal/` folders with `*_mask.png` files), from a Kaggle competition version of the dataset.
- The notebook downloads a zip from a Yandex Disk link (via `yadisk-direct`) and unzips it to `/content/content/kaggle_dataset_3/` with `train/` and `test/` folders.
- **The data is not included in this repo.**
- Labelled training data: 624 images (355 benign, 167 malignant, 102 normal), split 80/20 per class into 500 train / 124 validation. The 156 `test/` images have no masks; they are only used to make a Kaggle submission.
- Original dataset: Al-Dhabyani W., Gomaa M., Khaled H., Fahmy A. Dataset of Breast Ultrasound Images. *Data in Brief*, 2020; 28:104863. DOI: 10.1016/j.dib.2019.104863

## Approach

- Images are read in grayscale, resized to 256x256 and scaled to [0, 1]. Mask pixels inside the lesion get the class id (1 = benign, 2 = malignant); normal images are all background.
- Model (`ImprovedUNet`): 4-level U-Net (32 to 512 channels) with attention gates on the three deepest skip connections, 1 input channel, 3 output channels.
- Training (cell 12) runs two loops on the same model:
  1. 100 epochs with cross-entropy and Adam (lr 1e-3), logged to Weights & Biases.
  2. `train_multiclass_unet_improved`: 100 more epochs with a weighted mix of cross-entropy, Dice, focal and boundary losses, AdamW (lr 3e-4), cosine LR schedule and mixed precision, with class weights `[0.1, 1.0, 1.0]`.
- Evaluation: per-class Dice on the validation set each epoch, plus side-by-side plots of image / ground truth / prediction.
- Submission: predictions are resized back to the original image size and written as run-length encoded masks per class to `Improved_adv1.csv`.

## How to run

The notebook was written for Google Colab with a GPU.

1. `pip install -r requirements.txt`
2. Run the download cells, or put the data under `/content/content/kaggle_dataset_3/{train,test}` yourself (or change `root_dir` / `test_dir`).
3. Cell 12 calls `wandb.login()`, so you need a Weights & Biases account, or remove the W&B lines.
4. Run the cells in order.

## Results (from the committed outputs)

- The printed training log was cleared, so **no numeric metrics were recorded in the committed notebook**.
- The committed training plot (second loop) shows validation Dice that stays roughly flat at about 0.97 for background, 0.41 for benign and 0.20 for malignant, while validation loss rises over the 100 epochs. These values are read off the chart and the Dice averaging has a bug (see below), so treat them as rough.
- Cell 15 shows predictions for 20 validation images, with per-image Dice in each title. The two screenshots in the repo are examples of these plots.
- The submission cell wrote 156 rows, 26 of them with no predicted lesion. The Kaggle score is not recorded.

## Known limitations

- Data augmentation does nothing: `train_transform` is defined in cell 12 but never used, and the dataset's `transform` branch is just `pass`.
- Cell 12 trains twice on the same model (plain cross-entropy, then the combined loss), so the second loop starts from the first one's weights.
- The validation Dice average divides by the total number of batches, including batches where a class is absent, which pulls the benign and malignant scores down.
- The train/validation shuffle uses `random.shuffle` with no seed, and there is no labelled held-out test set.
- The RLE encoder (cell 17) drops a run that ends on the last pixel of the image.
- The binary-segmentation branch of `generate_submission` uses an undefined `original_sizes` (it is never reached with this 3-class model).
- Cell 15's output embeds about 4 MB of PNG images in the notebook.

## Unrelated notebook: `Final_Project.ipynb`

This notebook has nothing to do with cancer detection. It predicts the weekly close of the UNG (natural gas ETF) with a Keras LSTM (two LSTM layers of 50 units, 7-week window, training data up to 2020-05-24), retrains a small model on each 8-week window of the test period, and turns "predicted close > open" into a weekly buy signal.

- It needs `UNG_weekly_15y.csv`, which is **not in the repo**.
- Committed outputs: buy-signal direction accuracy 0.48 over 255 test weeks (precision 0.46, recall 0.51); average weekly return -0.0016 for buy-and-hold and -0.0055 for the strategy weeks.

Known issues:

- The return sign is flipped: cell 10 computes `Profit_Baseline = 100 *- Actual_Return`, so the profit curves are inverted.
- The Sharpe ratio (cell 13) divides annual return by `std / sqrt(52)` instead of `std * sqrt(52)`, which makes it about 52 times too large in magnitude.
- The buy threshold (cell 14) is tuned on the test set.
- The final `Dense(1, activation='relu')` layer cannot output negative values, but the targets are scaled to [-1, 1].
- Cell 16 calls `calculate_max_drawdown` and uses `Cumulative_Strategy_Optimized`, neither of which is defined in the notebook.
