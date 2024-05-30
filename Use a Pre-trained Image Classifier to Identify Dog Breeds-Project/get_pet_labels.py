#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_pet_labels.py
#                                                                             
# PROGRAMMER:red Hamami 
# DATE CREATED: 22/5/2024                                 
# REVISED DATE: 
# PURPOSE: Create the function get_pet_labels that creates the pet labels from 
#          the image's filename. This function inputs: 
#           - The Image Folder as image_dir within get_pet_labels function and 
#             as in_arg.dir for the function call within the main function. 
#          This function creates and returns the results dictionary as results_dic
#          within get_pet_labels function and as results within main. 
#          The results_dic dictionary has a 'key' that's the image filename and
#          a 'value' that's a list. This list will contain the following item
#          at index 0 : pet image label (string).
#
##
# Imports python modules
from os import listdir

# TODO 2: Define get_pet_labels function below please be certain to replace None
#       in the return statement with results_dic dictionary that you create 
#       with this function
# 
def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels based upon the filenames of the image files.
    The pet image labels are used to check the accuracy of the classifier function.

    Parameters:
     image_dir - The (full) path to the folder of images that are to be classified (string)
     
    Returns:
     results_dic - Dictionary with 'key' as image filename and 'value' as a List.
                   The list contains for following item:
                     index 0 = pet image label (string)
    """
    # Retrieve the filenames from folder pet_images/
    filename_list = listdir(image_dir)

    # Create an empty dictionary to store the results
    results_dic = dict()

    # Process each filename to create the pet image labels
    for filename in filename_list:
        # Skip files that start with a dot (like .DS_Store on macOS)
        if filename.startswith('.'):
            continue

        # Split the filename by '_' to break into words and convert to lowercase
        word_list_pet_image = filename.lower().split("_")

        # Create the pet name using list comprehension
        pet_name = " ".join([word.strip() for word in word_list_pet_image if word.isalpha()])

        # Add the filename and its corresponding label to the results dictionary
        results_dic[filename] = [pet_name]

    return results_dic