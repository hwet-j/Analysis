# GUI (윈도우 프로그래밍)
# wxPython 모듈 사용

import wx

# app = wx.App(False)
# frame = wx.Frame(None, wx.ID_ANY, 'Hi')
# frame.Show(True)
# app.MainLoop()

class Ex(wx.Frame):
    def __init__(self, parent, title):
        super(Ex, self).__init__(parent, title=title, size=(400, 300))
        
        # TextBox 추가
        # self.txtA = wx.TextCtrl(self)
        # self.txtA = wx.TextCtrl(self, style = wx.TE_MULTILINE)
        
        # 라벨과 텍스트박스 사용 
        panel = wx.Panel(self)
        wx.StaticText(panel, label = 'i d : ', pos = (5, 5))
        wx.StaticText(panel, label = 'pwd : ', pos = (5, 40))
        self.txtId = wx.TextCtrl(panel, pos = (40, 5))
        self.txtPwd = wx.TextCtrl(panel, pos = (40, 40),size = (200, -1))
        
        # 버튼 : 디자인
        btn1 = wx.Button(panel, label = '일반버튼1', pos = (5,100))
        btn2 = wx.Button(panel, label = '일반버튼2', pos = (100,100))
        btn3 = wx.Button(panel, label = '종료', pos = (200,100))
        
        # 버튼에 대한 이벤트 처리 작업1
        btn1.Bind(wx.EVT_BUTTON, self.btn1Method)
        btn2.Bind(wx.EVT_BUTTON, self.btn2Method)
        btn3.Bind(wx.EVT_BUTTON, self.btn3Method)
        
        # 버튼에 대한 이벤트 처리 작업2
        btn1.id = 1 # 각버튼을 구분하기 
        btn2.id = 2
        btn3.id = 3
        btn1.Bind(wx.EVT_BUTTON, self.OnClickMethod)
        btn2.Bind(wx.EVT_BUTTON, self.OnClickMethod)
        btn3.Bind(wx.EVT_BUTTON, self.OnClickMethod)
        
        self.Center()   # 모니터의 가운데에서 나오게
        self.Show()
        
    def abc(self):
        print('일반 메소드')
        
    def btn1Method(self, event):   # 이벤트 처리 메소드. event handler
        # print('버튼1을 누름')
        # id에 입력해준 값을 pwd에 출력
        imsi = self.txtId.GetValue()
        # print(imsi)
        self.txtPwd.SetLabelText(imsi)
        
    def btn2Method(self, event):   # 이벤트 처리 메소드. event handler
        # print('버튼2을 누름')
        msgDial = wx.MessageDialog(self, '메시지~', '제목!', wx.OK)
        msgDial.ShowModal()
        msgDial.Destroy()
        
    def btn3Method(self, event):   # 이벤트 처리 메소드. event handler
        #print('종료')
        self.Close() # wx.App()을 종료
    
    def OnClickMethod(self, event):
        # print('aa')
        # print(event.GetEventObject().id)
        if event.GetEventObject().id == 1:
            self.txtId.SetLabelText('kbs')
        elif event.GetEventObject().id == 2:
            self.txtId.SetLabelText('mbc')
        else:
            self.Close()
        
if __name__ == '__main__':
    app = wx.App()
    Ex(None, title = '연습')
    app.MainLoop()
    
