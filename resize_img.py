

from PIL import Image
import os
import sys

directory = 'deform/merge/'

for file_name in os.listdir(directory):
  print("Processing %s" % file_name)
  image = Image.open(os.path.join(directory, file_name))
  new_dimensions = (992, 992)
  output = image.resize(new_dimensions, Image.ANTIALIAS)

  output_file_name = os.path.join(directory, "small_" + file_name)
  output.save(output_file_name, "TIFF", quality = 95)

print("All done")