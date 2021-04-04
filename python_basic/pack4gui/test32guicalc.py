# -*- coding: utf-8 -*- 

###########################################################################
# # Python code generated with wxFormBuilder (version Jun 17 2015)
# # http://www.wxformbuilder.org/
# #
# # PLEASE DO "NOT" EDIT THIS FILE!
###########################################################################

import wx
import wx.xrc

###########################################################################
# # Class MyFrame1
###########################################################################


class MyFrame1 (wx.Frame):
    
    def __init__(self, parent):
        wx.Frame.__init__ (self, parent, id=wx.ID_ANY, title=u"간단 계산기", pos=wx.DefaultPosition, size=wx.Size(320, 363), style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL)
        
        # self.SetSizeHintsSz( wx.DefaultSize, wx.DefaultSize )
        self.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_MENU))
        
        bSizer6 = wx.BoxSizer(wx.VERTICAL)
        
        self.m_panel4 = wx.Panel(self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL)
        bSizer7 = wx.BoxSizer(wx.HORIZONTAL)
        
        self.m_staticText6 = wx.StaticText(self.m_panel4, wx.ID_ANY, u"숫자1 : ", wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_staticText6.Wrap(-1)
        bSizer7.Add(self.m_staticText6, 0, wx.ALL, 5)
        
        self.txtNum1 = wx.TextCtrl(self.m_panel4, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0)
        bSizer7.Add(self.txtNum1, 0, wx.ALL, 5)
        
        self.m_panel4.SetSizer(bSizer7)
        self.m_panel4.Layout()
        bSizer7.Fit(self.m_panel4)
        bSizer6.Add(self.m_panel4, 0, wx.EXPAND | wx.ALL, 5)
        
        self.m_panel5 = wx.Panel(self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL)
        bSizer8 = wx.BoxSizer(wx.HORIZONTAL)
        
        self.m_staticText9 = wx.StaticText(self.m_panel5, wx.ID_ANY, u"숫자2 : ", wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_staticText9.Wrap(-1)
        bSizer8.Add(self.m_staticText9, 0, wx.ALL, 5)
        
        self.txtNum2 = wx.TextCtrl(self.m_panel5, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0)
        bSizer8.Add(self.txtNum2, 0, wx.ALL, 5)
        
        self.m_panel5.SetSizer(bSizer8)
        self.m_panel5.Layout()
        bSizer8.Fit(self.m_panel5)
        bSizer6.Add(self.m_panel5, 0, wx.EXPAND | wx.ALL, 5)
        
        self.m_panel6 = wx.Panel(self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL)
        bSizer9 = wx.BoxSizer(wx.HORIZONTAL)
        
        rdoOpChoices = [ u"+", u"*", u"-", u"/" ]
        self.rdoOp = wx.RadioBox(self.m_panel6, wx.ID_ANY, u"연산자 선택", wx.DefaultPosition, wx.DefaultSize, rdoOpChoices, 1, wx.RA_SPECIFY_ROWS)
        self.rdoOp.SetSelection(0)
        bSizer9.Add(self.rdoOp, 1, wx.ALL, 5)
        
        self.m_panel6.SetSizer(bSizer9)
        self.m_panel6.Layout()
        bSizer9.Fit(self.m_panel6)
        bSizer6.Add(self.m_panel6, 0, wx.EXPAND | wx.ALL, 5)
        
        self.m_panel7 = wx.Panel(self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL)
        bSizer10 = wx.BoxSizer(wx.HORIZONTAL)
        
        self.m_staticText10 = wx.StaticText(self.m_panel7, wx.ID_ANY, u"연산결과 : ", wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_staticText10.Wrap(-1)
        bSizer10.Add(self.m_staticText10, 0, wx.ALL, 5)
        
        self.staResult = wx.StaticText(self.m_panel7, wx.ID_ANY, u"0", wx.DefaultPosition, wx.DefaultSize, 0)
        self.staResult.Wrap(-1)
        bSizer10.Add(self.staResult, 1, wx.ALL, 5)
        
        self.m_panel7.SetSizer(bSizer10)
        self.m_panel7.Layout()
        bSizer10.Fit(self.m_panel7)
        bSizer6.Add(self.m_panel7, 0, wx.EXPAND | wx.ALL, 5)
        
        self.m_panel8 = wx.Panel(self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL)
        bSizer11 = wx.BoxSizer(wx.HORIZONTAL)
        
        self.btnCalc = wx.Button(self.m_panel8, wx.ID_ANY, u"계산", wx.DefaultPosition, wx.DefaultSize, 0)
        bSizer11.Add(self.btnCalc, 0, wx.ALL, 5)
        
        self.btnClear = wx.Button(self.m_panel8, wx.ID_ANY, u"초기화", wx.DefaultPosition, wx.DefaultSize, 0)
        bSizer11.Add(self.btnClear, 0, wx.ALL, 5)
        
        self.btnExit = wx.Button(self.m_panel8, wx.ID_ANY, u"종료", wx.DefaultPosition, wx.DefaultSize, 0)
        bSizer11.Add(self.btnExit, 0, wx.ALL, 5)
        
        self.m_panel8.SetSizer(bSizer11)
        self.m_panel8.Layout()
        bSizer11.Fit(self.m_panel8)
        bSizer6.Add(self.m_panel8, 0, wx.EXPAND | wx.ALL, 5)
        
        self.SetSizer(bSizer6)
        self.Layout()
        
        self.Centre(wx.BOTH)
        
        # Connect Events
        self.btnCalc.id = 1
        self.btnClear.id = 2
        self.btnExit.id = 3
        self.btnCalc.Bind(wx.EVT_BUTTON, self.OnCalcProcess)
        self.btnClear.Bind(wx.EVT_BUTTON, self.OnCalcProcess)
        self.btnExit.Bind(wx.EVT_BUTTON, self.OnCalcProcess)
    
    def __del__(self):
        pass
    
    # Virtual event handlers, overide them in your derived class
    def OnCalcProcess(self, event):
        sel_id = event.GetEventObject().id
        # print(sel_id)
        if sel_id == 1:
            op = self.rdoOp.GetStringSelection()
            # print(op)
            num1 = self.txtNum1.GetValue()
            num2 = self.txtNum2.GetValue()
            
            if num1 == '' or num2 == '':
                wx.MessageBox('값을 입력하시요', '알림', wx.OK)
                return
            try:
                result = eval(num1 + op + num2)
            except Exception as e:
                wx.MessageBox('연산 오류 :' + str(e), '알림', wx.OK)
                return
            
            self.staResult.SetLabel(str(result))
            
        elif sel_id == 2:
            self.txtNum1.SetLabel('')   # 값을 비워줌
            self.txtNum2.SetLabel('')
            self.staResult.SetLabel('')
            self.rdoOp.SetSelection(0)  # 값을(radio) 첫번째 (0)값으로 설정
            self.txtNum1.SetFocus() # UX/UI
            
        elif sel_id == 3:
            dlg = wx.MessageDialog(self, '정말 종료할까요?', '알림', wx.YES_NO)
            imsi = dlg.ShowModal()
            if imsi == wx.ID_YES:
                dlg.Destroy()   # MessageDaialog 닫기
                self.Close()    # Frame 닫기
            
        
    
    
if __name__ == '__main__':
    app = wx.App()
    MyFrame1(None).Show()
    app.MainLoop()
