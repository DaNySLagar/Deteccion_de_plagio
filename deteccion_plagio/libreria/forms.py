from django import forms
from multiupload.fields import MultiFileField


class MiFormulario(forms.Form):
    opciones = [("1", "Detectar Plagio"), ("0", "Detectar Similitud")]
    seleccion = forms.ChoiceField(choices=opciones, widget=forms.RadioSelect)
    documentos = MultiFileField(min_num=1, max_num=20, required=False)
    texto = forms.CharField(widget=forms.Textarea, required=False)

