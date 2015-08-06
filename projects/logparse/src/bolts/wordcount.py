from __future__ import absolute_import, print_function, unicode_literals

from collections import Counter
from streamparse.bolt import Bolt
import smtplib
from email.mime.text import MIMEText

class WordCounter(Bolt):

    def initialize(self, conf, ctx):
        nothing = 5

    def process(self, tup):
        msg = MIMEText('ERROR')
        msg['subject'] = 'Testing error'
        msg['From'] = 'noreply@caringbridge.org'
        msg['To'] = 'dmurali@caringbridge.org'
        mailSrv = smtplib.SMTP('localhost', 1025)
        line = tup.values[0]
        if 'ERROR' in set(line.split()):
            mailSrv.sendmail('noreply@caringbridge.org', ['dmurali@caringbridge.org'], msg.as_string())
        mailSrv.quit()

