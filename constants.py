FQ_START_INDEX = 16
FEATURES_NUM = 16 + 16

FQ_MEANS = ['125_250BA', '500_1000BA',
       '1500_2000BA', '3000_4000BA', '6000_8000BA', '500_1000BB',
       '1500_2000BB', '3000_4000BB']
FQ_FEATURES = ['125BefAir', '250BefAir',
       '500BefAir', '1000BefAir', '1500BefAir', '2000BefAir', '3000BefAir',
       '4000BefAir', '6000BefAir', '8000BefAir', '500BefBone', '1000BefBone',
       '1500BefBone', '2000BefBone', '3000BefBone', '4000BefBone']
BASE_FEATURES = ['Gender-0=M, 1 =F', 'Age', 'Side 0=L, 1=R']
CONSTRUCTED_FEATURES = ['Air mean',
       'Bone mean', 'Bone/Air', 'Bone*Air','ABG']

CLASSES = ['125AftAir', '250AftAir',
       '500AftAir', '1000AftAir', '1500AftAir', '2000AftAir', '3000AftAir',
       '4000AftAir', '6000AftAir', '8000AftAir', '500AftBone', '1000AftBone',
       '1500AftBone', '2000AftBone', '3000AftBone', '4000AftBone']
