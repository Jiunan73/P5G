from wtforms import Form, SelectField, SubmitField
from wtforms.validators import DataRequired


class TargetForm(Form):
    target_pan = SelectField(label="Pan:", validators=[DataRequired
                                                       ])
    target_tilt = SelectField(label="Tilt:", validators=[DataRequired
                                                         ])
    target_zoom = SelectField(label="Zoom:", validators=[DataRequired
                                                         ])
    submit = SubmitField(label="提交")
