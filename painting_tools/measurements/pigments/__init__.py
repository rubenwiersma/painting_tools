from .pigment_database import PigmentDatabase

cmt = PigmentDatabase('cmt')
oilpaint = PigmentDatabase('oilpaint')
curtis = PigmentDatabase('curtis')
basic = PigmentDatabase('basic')

cmy, _ = basic.get_pigments('cmy')
cyan = cmy[0]
magenta = cmy[1]
yellow = cmy[2]