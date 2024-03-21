import tensorflow as tf
import numpy as np
import random
import os

class MapFunction:
	def __init__(self, imageSize):
		# define the image width and height
		self.imageSize = imageSize

	def decode_and_resize(self, imagePath):
		# read and decode the image path
		image = tf.io.read_file(imagePath)
		image = tf.image.decode_jpg(image, channels=3)
		# convert the image data type from uint8 to float32 and then resize
		# the image to the set image size
		image = tf.image.convert_image_dtype(image, dtype=tf.float32)
		# image = tf.image.resize(image, self.imageSize)
		# return the image
		return image

	def __call__(self, anchor, positive, negative):
		anchor = self.decode_and_resize(anchor)
		positive = self.decode_and_resize(positive)
		negative = self.decode_and_resize(negative)
		# return the anchor, positive and negative processed images
		return anchor, positive, negative


class TripletGenerator:
	def __init__(self, datasetPath):
		# create an empty list which will contain the subdirectory
		# names of the `dataset` directory with more than one image
		# in it
		self.fruitNames = list()
		# iterate over the subdirectories in the dataset directory
		for folderName in os.listdir(datasetPath):
			# build the subdirectory name
			absoluteFolderName = os.path.join(datasetPath, folderName)
			# get the number of images in the subdirectory
			numImages = len(os.listdir(absoluteFolderName))
			# if the number of images in the current subdirectory
			# is more than one, append into the `fruitNames` list
			if numImages > 1:
				self.fruitNames.append(absoluteFolderName)
		# create a dictionary of people name to their image names
		self.allFruits = self.generate_all_fruits_dict()

	def generate_all_fruits_dict(self):
		# create an empty dictionary that will be populated with
		# directory names as keys and image names as values
		allFruits = dict()
		# iterate over all the directory names with more than one
		# image in it
		for fruit in self.fruitNames:
			# get all the image names in the current directory
			imageNames = os.listdir(fruit)
			# build the image paths and populate the dictionary
			fruitPhotos = [
				os.path.join(fruit, imageName) for imageName in imageNames
			]
			allFruits[fruit] = fruitPhotos
		# return the dictionary
		return allFruits

	def get_next_element(self):
		# create an infinite generator
		while True:
			# draw a person at random which will be our anchor and
			# positive person
			anchorName = random.choice(self.fruitNames)
			# copy the list of people names and remove the anchor
			# from the list
			temporaryNames = self.fruitNames.copy()
			temporaryNames.remove(anchorName)
			# draw a person at random from the list of people without
			# the anchor, which will act as our negative sample
			negativeName = random.choice(temporaryNames)
			# draw two images from the anchor folder without replacement
			(anchorPhoto, positivePhoto) = np.random.choice(
				a=self.allFruits[anchorName],
				size=2,
				replace=False
			)
			# draw an image from the negative folder
			negativePhoto = random.choice(self.allFruits[negativeName])
			# yield the anchor, positive and negative photos
			yield anchorPhoto, positivePhoto, negativePhoto
