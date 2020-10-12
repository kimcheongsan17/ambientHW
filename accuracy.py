# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Example using TF Lite to classify a given image using an Edge TPU.
   To run this code, you must attach an Edge TPU attached to the host and
   install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
   device setup instructions, see g.co/coral/setup.
   Example usage (use `install_requirements.sh` to get these files):
   ```
   python3 classify_image.py \
     --model models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite  \
     --labels models/inat_bird_labels.txt \
     --input images/parrot.jpg
   ```
"""

import argparse
import time

from PIL import Image

import classify
import tflite_runtime.interpreter as tflite
import platform
import os

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]


def load_labels(path, encoding='utf-8'):
  """Loads labels from file (with or without index numbers).
  Args:
    path: path to label file.
    encoding: label file encoding.
  Returns:
    Dictionary mapping indices to labels.
  """
  with open(path, 'r', encoding=encoding) as f:
    lines = f.readlines()
    if not lines:
      return {}

    if lines[0].split(' ', maxsplit=1)[0].isdigit():
      pairs = [line.split(' ', maxsplit=1) for line in lines]
      return {int(index): label.strip() for index, label in pairs}
    else:
      return {index: line.strip() for index, line in enumerate(lines)}


def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '-m', '--model', required=True, help='File path of .tflite file.')
  parser.add_argument(
      '-i', '--image_folder', help='Folder path of image and label.')
  parser.add_argument(
      '-t', '--threshold', type=float, default=0.0,
      help='Classification score threshold')
  parser.add_argument(
      '-c', '--count', type=int, default=5,
      help='Number of times to run inference')
  args = parser.parse_args()

  interpreter = make_interpreter(args.model)
  interpreter.allocate_tensors()


  size = classify.input_size(interpreter)
  with open(os.path.join(args.image_folder, 'labels.txt'),'r') as l:
    answer=l.readlines()
    for i,a in enumerate(answer):
      answer[i]=a.rstrip()

  total=0
  correct=0

  for (root, dirs, files) in os.walk(args.image_folder):
    for file in files:
      if file!='labels.txt':
        image_name=file.split('.')[0]

        image = Image.open(os.path.join(root, file)).convert('RGB').resize(size, Image.ANTIALIAS)
        classify.set_input(interpreter, image)

        for _ in range(args.count):
          interpreter.invoke()
          classes = classify.get_output(interpreter, 1, args.threshold)
        for klass in classes:
          if int(klass.id)==int(answer[int(image_name)]):
            correct+=1
            print("Correct")
          else:
            print("Wrong")

        total+=1


  print('-------RESULTS--------')
  print("Acurracy: ", correct/total)



if __name__ == '__main__':
  main()