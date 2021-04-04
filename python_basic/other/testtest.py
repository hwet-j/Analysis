class ElecProduct:
    volume = 0
    def volumeControl(self, volume):
        print('부모 클래스')

class ElecTv(ElecProduct):
    def volumeControl(self, volume):
        print('Tv의 볼륨 조절')
        
class ElecRadio(ElecProduct):
    def volumeControl(self, volume):
        print('Radio의 볼륨 조절')

electv = ElecTv()
electv.volumeControl(40)

elecradio = ElecRadio()
elecradio.volumeControl(40)