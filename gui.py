import wx

from ner_model import NER_Model


class GUI(wx.Frame):
    def __init__(self, parent, _model):
        wx.Frame.__init__(self, parent, title='RO NER')
        self.Maximize(True)
        self.model = _model
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.panel = wx.Panel(self)

        self.label_text = wx.StaticText(self.panel, label="Text:")
        self.sizer.Add(self.label_text, 0, wx.ALL | wx.EXPAND, 5)

        self.input_text = wx.TextCtrl(self.panel)
        self.sizer.Add(self.input_text, 0, wx.ALL | wx.EXPAND, 5)

        self.button = wx.Button(self.panel, label="Check")
        self.button.Bind(wx.EVT_BUTTON, self.on_click)
        self.sizer.Add(self.button, 0, wx.ALL | wx.CENTER, 5)

        self.label_interp = wx.StaticText(self.panel, label="Interpretation:")
        self.sizer.Add(self.label_interp, 0, wx.ALL | wx.LEFT, 5)

        self.label_result = wx.StaticText(self.panel, label="", style=0)
        self.sizer.Add(self.label_result, 0, wx.ALL | wx.LEFT, 5)

        self.panel.SetSizer(self.sizer)

    def on_click(self, e):
        text = self.model.predict(self.input_text.GetValue())
        self.label_result.SetLabelMarkup(text)


if __name__ == '__main__':
    app = wx.App(False)
    model = NER_Model()
    frame = GUI(None, model)
    frame.Show()
    app.MainLoop()
