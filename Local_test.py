#!/zhome/7c/0/155777/anaconda3/envs/ml/bin/python

from Main_Training import *
from Functions.Scoring_Metrics import *
from Functions.VisualizeMasks import *
from datetime import date
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Train part
    parser = ArgumentParser(description="Training script for segmentation models")
    # Data paths
    parser.add_argument("-image", action="store", dest="image_dir_name", type=str, default='images_correct_labels' , help="Directory containing images")
    parser.add_argument("-label", action="store", dest="label_dir_name", type=str, default='mask_images_correct_labels' , help="Directory containing image labels")
    
    model_list=["fcn",
                "unet",
                "unet_pretrained",
                "dlv3",
                "dlv3p"]
    
    parser.add_argument("-model", action="store", dest="model_pick", type=str, default='dlv3',
                        help=f'Chose a model type from these; {model_list}')

    parser.add_argument("-model_path", action="store", dest="model_path", type=str, default='Data/output_tl/UNet_resnet50_pretrained_sorted_classes_2024-12-12.pth',
                        help="specify path to model")

    parser.add_argument("-n", action="store", dest="num_images", type=int, default=20,
                        help="Number of test images")
    
    parser.add_argument("-suffix", action="store", dest="name_suffix", type=str, default=None,
                        help="Append given string to output image name")

    parser.add_argument("-dice",action="store", dest="dice", type=str2bool, default=False,
                        help="Test all test images with Dice scores. Default: False")

    parser.add_argument("-pred_img",action="store", dest="pred_img", type=str2bool, default=True,
                        help="If set to true the function will make an image of the predictions and save to a file. Default: True")


    args = parser.parse_args()
    image_dir_name = args.image_dir_name
    label_dir_name = args.label_dir_name
    model_pick = args.model_pick
    model_path = args.model_path
    num_images = args.num_images
    name_suffix = args.name_suffix
    dice = args.dice
    pred_img = args.pred_img

    print(f"dice={dice}")
    print(f"pred_img={pred_img}")

    # Setup directories
    data_dir = os.path.join('/zhome/7c/0/155777/Desktop/Code/Data')
    output_dir = os.path.join(data_dir, "output")
    image_dir = os.path.join(data_dir, image_dir_name)
    label_dir = os.path.join(data_dir, label_dir_name)


    model_name = model_path.split("/")[-1].split(".")[-2]

    # setting up datasets
    dataset = CarcassDataset(image_dir, label_dir)
    split_datasets = train_val_dataset(dataset)

    num_classes=6


    if model_pick.lower() == 'fcn':
        model_name="FCN_custom_sorted_classes"
        model=FCN(num_classes)
        model.load_state_dict(torch.load(model_path,weights_only=True, map_location=torch.device('cpu')))

    elif model_pick.lower() == 'unet':
        # Setup model
        model_name="UNet_custom_sorted_classes"
        model = UNet(num_classes)
        model.load_state_dict(torch.load(model_path,weights_only=True, map_location=torch.device('cpu')))

    elif model_pick.lower() == 'unet_pretrained':
        # Setup model
        model_name="UNet_resnet50_pretrained_sorted_classes"
        model = smp.Unet('resnet50',in_channels=3, encoder_weights='imagenet',classes=num_classes, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16])
        model.load_state_dict(torch.load(model_path,weights_only=True, map_location=torch.device('cpu')))

    elif model_pick.lower() == 'dlv3':
        # Setup model
        model_name="deeplab_v3_resnet50_pretrained_sorted_classes"
        model = smp.DeepLabV3('resnet50',in_channels=3, encoder_weights='imagenet',classes=num_classes)
        model.load_state_dict(torch.load(model_path,weights_only=True, map_location=torch.device('cpu')))
    
    elif model_pick.lower() == 'dlv3p':
        # Setup model
        model_name="DeepLabV3Plus_resnet50_pretrained_sorted_classes"
        model = smp.DeepLabV3Plus('resnet50',in_channels=3, encoder_weights='imagenet',classes=num_classes)
        model.load_state_dict(torch.load(model_path,weights_only=True, map_location=torch.device('cpu')))

    else:
        raise ValueError(f"{model_pick} in not not in {model_list}")
    
    if name_suffix is not None:
        model_name=model_name+'_'+name_suffix

    # Print status
    print(f"{model_path} loaded as {model_pick}.")

    test_batch_size = 5
    data_loader = DataLoader(split_datasets["val"], batch_size=test_batch_size)
    
    if pred_img:
        data_iter= iter(data_loader)
        
        fig, axes = plt.subplots(num_images, 4, figsize=(3*5, num_images*4)) # setup plot

        print(f"Making predictions for {num_images} test images:\n")
        for j in tqdm(range(int(num_images/test_batch_size))):
            X, Y = next(data_iter)

            with torch.no_grad():
                X, Y = X.to(device), Y.to(device)
                Y_pred = model(X)
                Y_pred = torch.argmax(Y_pred, dim=1)

            for i in range(test_batch_size):

                i_plot=i+test_batch_size*j

                image = X[i].permute(1, 2, 0).cpu().detach().numpy().astype(int)  # make the image from a tensor to skimage-type np array
                label_class = Y[i].cpu().detach().numpy()
                label_class_predicted = Y_pred[i].cpu().detach().numpy()
                applied_mask_img=image.copy()
                applied_mask_img[label_class_predicted==0]=0

                axes[i_plot, 0].imshow(image)
                axes[i_plot, 0].set_title("Image")
                axes[i_plot, 1].imshow(draw_masks(image,label_class))
                axes[i_plot, 1].set_title("True label")
                axes[i_plot, 2].imshow(draw_masks(image,label_class_predicted))
                axes[i_plot, 2].set_title("Predicted")
                #axes[i, 3].imshow(label2rgb(label_class_predicted - label_class))
                #axes[i, 3].set_title("Difference")
                axes[i_plot, 3].imshow(applied_mask_img)
                axes[i_plot, 3].set_title("Applied mask")

        fig.savefig(f"{output_dir}/{model_name}_test_images_{date.today()}.png")   # save images
    

    ##### Dice testing #####
    if dice:
        print("Beginning Dice testing:\n")
        all_scores=dice_validation(model, data_loader, device, num_classes)  # run test from Scoring Metrics
        mean_scores=np.nanmean(all_scores,axis=0)

        np.save(output_dir+f"/{model_name}_Raw_dice_{date.today()}",all_scores)  # save all scores to np.array-file
        with open(output_dir+f"/{model_name}_dice_results_{date.today()}.txt","w") as f:  # Save mean scores per class to a .txt
            for c in range(num_classes):
                line=f"Class_{c}:\t{mean_scores[c]}"
                f.write(line+"\n")
                print(line)