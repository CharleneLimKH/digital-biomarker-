import numpy as np 
import matplotlib.pyplot as plt 

def get_specific_data(folder_path, tumor_class, indices_list):
    
    us_image = list()
    seg_masks = list()
    
    for index in indices_list:
        img_file_name = f"{tumor_class} ({index}).png"
        mask_file_name = f"{tumor_class} ({index})_mask.png"
        
        us_image.append(plt.imread(f"{folder_path}/{tumor_class}/{img_file_name}")[:,:,0])
        seg_masks.append(plt.imread(f"{folder_path}/{tumor_class}/{mask_file_name}"))
    
    
    return us_image, seg_masks


def get_benign_data_sitk(folder_path, imgs_index):
    import SimpleITK as sitk
    img_file_name = f"benign ({imgs_index}).png"
    mask_file_name = f"benign ({imgs_index})_mask.png"
        

    us_image = sitk.ReadImage(f"{folder_path}/benign/{img_file_name}", sitk.sitkInt16)
    seg_mask = sitk.ReadImage(f"{folder_path}/benign/{mask_file_name}", sitk.sitkInt16)

    return us_image, seg_mask