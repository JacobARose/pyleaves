# Created 11/13/2020
# Author: Jacob A Rose




# TODO: Finish altering these 2 functions for visualizing a selection of images along with their true label and incorrectly predicted label
# TODO: Also create a new function to visualize individual images with their top-5 predictions, along with the percent confidence/probability.
# import matplotlib.pyplot as plt


def print_trainable_layers(model):
    """Display each layer's name along with True or False for its trainable attribute.

    Args:
        model ([type]): [description]
    """    
    for layer in model.layers:
        print("{0}:\t{1}".format(layer.trainable, layer.name))



# # Grab random images from the test and make predictions using
# # the model *while it is training* and log them using WnB
# def get_sample_predictions():
#    predictions = []
#    images = []
#    random_indices = np.random.choice(X_test.shape[0], 25)
#    for index in random_indices:
#        image = X_test[index].reshape(1, 28, 28, 1)
#        prediction = np.argmax(model(image).numpy(), axis=1)
#        prediction = CLASSES[int(prediction)]
       
#        images.append(image)
#        predictions.append(prediction)
   
#    wandb.log({"predictions": [wandb.Image(image, caption=prediction)
#                               for (image, prediction) in zip(images, predictions)]})        


# def plot_images(images, cls_true, cls_pred=None, smooth=True):
# """
# Source code: https://colab.research.google.com/github/Hvass-Labs/TensorFlow-Tutorials/blob/master/10_Fine-Tuning.ipynb
# """
#     assert len(images) == len(cls_true)

#     # Create figure with sub-plots.
#     fig, axes = plt.subplots(3, 3)

#     # Adjust vertical spacing.
#     if cls_pred is None:
#         hspace = 0.3
#     else:
#         hspace = 0.6
#     fig.subplots_adjust(hspace=hspace, wspace=0.3)

#     # Interpolation type.
#     if smooth:
#         interpolation = 'spline16'
#     else:
#         interpolation = 'nearest'

#     for i, ax in enumerate(axes.flat):
#         # There may be less than 9 images, ensure it doesn't crash.
#         if i < len(images):
#             # Plot image.
#             ax.imshow(images[i],
#                       interpolation=interpolation)

#             # Name of the true class.
#             cls_true_name = class_names[cls_true[i]]

#             # Show true and predicted classes.
#             if cls_pred is None:
#                 xlabel = "True: {0}".format(cls_true_name)
#             else:
#                 # Name of the predicted class.
#                 cls_pred_name = class_names[cls_pred[i]]

#                 xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

#             # Show the classes as the label on the x-axis.
#             ax.set_xlabel(xlabel)
        
#         # Remove ticks from the plot.
#         ax.set_xticks([])
#         ax.set_yticks([])
    
#     # Ensure the plot is shown correctly with multiple plots
#     # in a single Notebook cell.
#     plt.show()


# def plot_example_errors(cls_pred):
#     # cls_pred is an array of the predicted class-number for
#     # all images in the test-set.

#     # Boolean array whether the predicted class is incorrect.
#     incorrect = (cls_pred != cls_test)

#     # Get the file-paths for images that were incorrectly classified.
#     image_paths = np.array(image_paths_test)[incorrect]

#     # Load the first 9 images.
#     images = load_images(image_paths=image_paths[0:9])
    
#     # Get the predicted classes for those images.
#     cls_pred = cls_pred[incorrect]

#     # Get the true classes for those images.
#     cls_true = cls_test[incorrect]
    
#     # Plot the 9 images we have loaded and their corresponding classes.
#     # We have only loaded 9 images so there is no need to slice those again.
#     plot_images(images=images,
#                 cls_true=cls_true[0:9],
#                 cls_pred=cls_pred[0:9])