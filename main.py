import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
from mss import mss
from PIL import Image
from PIL import Image
import pandas as pd
import pickle
from sklearn.externals import joblib
from fastai2.vision.all import *
from fastai2.vision.widgets import *
from fastai2.basics import *
import mouse as m
import time as t
import random as r


"""
Configuration instructions

- Install most librairies : python -m pip install -r requirements.txt, python -m pip install opencv-python, pillow
- follow the following tutorial : https://docs.fast.ai/install.html
- Change the model's path : path
- Go to your python file and find pathlib.py and replace yours with the one in the current folder
- Now place your emulator on your main screen
- Try resizing bluestack until the program work
- Enjoy your free time farming :)
"""


# --------------------------------------------------------------------------   Configuration and settings   -----------------------------------------------------------------------#


path = Path('D:/Dev/Python/Test - Bot/final2.pkl') # Path to get the Ml model
imagePath = Path('./Templates/') # Path to get images
frameRateNonAnalysis = 20 # Programm will analyse 1 frame of 20
bounding_box = {'top': 0, 'left': 0, 'width': 1800, 'height': 1200} # Screen configuration | if you are using a second screen modify this value
averageMouseDeplacement = 1 # Allows the programm to move the mouse | time in second
averageWaitingTime = 1 # Allows the programm to wait in order to net get ban
Refill = 1 # 1 = yes / 2 = no
TypeName = 'Dungeon' # 'Dungeon' / 'Rifts'






# ----------------------------------------------------------------------------   Functions and classes   --------------------------------------------------------------------------#

def verif(array):
	for num in array:
		for numb in array:
			if num != numb:
				return False
	return True

def grand_parent_labeler(o):
    "Label `item` with the parent folder name."
    return Path(o).parent.name

def checkTemplate(screen, template, order):
	global c
	w, h = template['image'].shape[::-1]

	# All the 6 methods for comparison in a list
	methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
	            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

	result = cv2.matchTemplate(screen, template['image'], cv2.TM_SQDIFF)
	(minVal, maxVal, minLocation, maxLocation) = cv2.minMaxLoc(result)
	# print(minVal, template['name'])

	if (minVal <= 2*template['match_score'] and order[c][2] != 2):
		global bestLocation
		bestLocation = (int(minLocation[0] + template['image'].shape[1]/2), int(minLocation[1] + template['image'].shape[0]/2))
		# cv2.circle(screen, (bestLocation),50, (0, 255, 0), 2)
		# cv2.imshow('test', screen)
		# cv2.waitKey(0)
		clickOn(order[c])
		c += 1
	elif order[c][2] == 0:
		c += 1

	if c == 4:
		forcePathFail = False

	if c == len(order):
		c = 0
		endAction = True

def clickOn(label):
	global averageWaitingTime
	global averageMouseDeplacement
	for i in range(label[1]):
		duration = r.randrange(0,averageMouseDeplacement)/2
		m.move(bestLocation[0] + ui_template[label[0]]['image'].shape[1] * r.randrange(-1, 1)/2, bestLocation[1] + ui_template[label[0]]['image'].shape[0] * r.randrange(-1, 1)/2, duration=duration)
		m.click(button='left')
		sleep(averageWaitingTime + duration)


# -----------------------------------------------------------------------------   Variables setup   ---------------------------------------------------------------------------#


typeRun = {
	'Dungeon' : [['Victory', 3, 1], ['Cross', 1, 1]],
	'Rifts' : [['Damage', 5, 1], ['Cross-Rift', 1, 1]]
}

bestLocation = (0,0)
ui_template = {
				'Rejouer' : {'image':cv2.imread(str(imagePath/'Rejouer.jpg'),0), 'match_score':9037656.0, 'name':'Rejouer'},
				'Cross' : {'image':cv2.imread(str(imagePath/'Cross.jpg'),0), 'match_score':100000.0, 'name':'Cross'},
				'Oui' : {'image':cv2.imread(str(imagePath/'Oui.jpg'),0), 'match_score':6000312.0, 'name':'Oui'},
				'Fermer' : {'image':cv2.imread(str(imagePath/'Fermer.jpg'),0), 'match_score':3665168.0, 'name':'Fermer'},
				'+90' : {'image':cv2.imread(str(imagePath/'+90.jpg'),0), 'match_score':3532480.0, 'name':'+90'},
				'Ok' : {'image':cv2.imread(str(imagePath/'Ok.jpg'),0), 'match_score':1445952.0, 'name':'Ok'},
				'Shop' : {'image':cv2.imread(str(imagePath/'Shop.jpg'),0), 'match_score':3532480.0, 'name':'Shop'},
				'Non' : {'image':cv2.imread(str(imagePath/'Non.jpg'),0), 'match_score':200056.0, 'name':'Non'},
				'Preparation' : {'image':cv2.imread(str(imagePath/'Preparation.jpg'),0), 'match_score':400056.0, 'name':'Preparation'},
				'Go' : {'image':cv2.imread(str(imagePath/'Go.jpg'),0), 'match_score':1800056.0, 'name':'Go'},
				'Victory' : {'image':cv2.imread(str(imagePath/'Victory.jpg'),0), 'match_score':152788.0/1.9, 'name':'Victory'},
				'Damage' : {'image':cv2.imread(str(imagePath/'Damage.jpg'),0), 'match_score':152788.0/1.9, 'name':'Damage'},
				'Cross-Rift' : {'image':cv2.imread(str(imagePath/'Cross-Rift.jpg'),0), 'match_score':152788.0/1.9, 'name':'Cross-Rift'}
			  }


orderSuccess = [
					typeRun[TypeName][0],
					typeRun[TypeName][1],
					['Cross', 1, 0],
					['Rejouer', 1, 1],
					['Shop', 1, Refill],
					['+90', 1, 1],
					['Oui', 1, 1],
					['Ok', 1, 1],
					['Fermer', 1, 1],
					['Rejouer', 1, 1]
			   ]
orderFail = [
					['Non', 1, 1],
					['Victory', 1, 1],
					['Preparation', 1, 1],
					['Go', 1, 1],
					['Shop', 1, Refill],
					['+90', 1, 1],
					['Oui', 1, 1],
					['Ok', 1, 1],
					['Fermer', 1, 1],
					['Rejouer', 1, 1],
					['Go', 1, 1]
			]

sct = mss()
learn_inf = load_learner(path)
actu_pred = False
prev_preds = []

i = frameRateNonAnalysis
c = 2
forcePath = False
endAction = False
nbRun = 0


# -------------------------------------------------------------------------------   Main Section   -----------------------------------------------------------------------------#



while True:
	if(i == frameRateNonAnalysis):
		sct_img = sct.grab(bounding_box)
		output = np.array(Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX"))
		pred,pred_idx,probs = learn_inf.predict(output)
		print('nombre de run : ' + str(nbRun), end = '\r')
		# print(pred)
		if len(prev_preds) > 3:
			prev_preds.insert(0, pred)
			prev_preds.pop()
		else:
			prev_preds.append(pred)

		if pred == 'Wave' or pred == 'Boss':
			c = 0
			endAction = False

		if verif(prev_preds):
			if not(endAction):
				if prev_preds[0] == 'Fail' or forcePath == 'Fail':
					if c < len(orderFail):
						checkTemplate(cv2.cvtColor(output,cv2.COLOR_RGB2GRAY), ui_template[orderFail[c][0]], orderFail)
						forcePathFail = 'Fail'
				if prev_preds[0] == 'Reward' or forcePath == 'Success':
					# for template in ui_template:
					# 	checkTemplate(cv2.cvtColor(output,cv2.COLOR_RGB2GRAY), ui_template[template])
					if c < len(orderSuccess):
						if c == 3:
							nbRun += 1
						checkTemplate(cv2.cvtColor(output,cv2.COLOR_RGB2GRAY), ui_template[orderSuccess[c][0]], orderSuccess)
						forcePathFail = 'Success'
		i = 0
	i+=1

	if (cv2.waitKey(1) & 0xFF) == ord('q'):
		cv2.destroyAllWindows()