#!/usr/bin/env python
#title           :mail.py
#description     :Send an email notification to specified recervier via GOOGLE gmail.
#author          :siyu zhu
#date            :Nov 30th, 2014
#version         :0.1
#usage           :from mail import mail
#notes           :
#python_version  :2.7

def mail(receiver,Message):
    	import smtplib
	import email.utils
	from email.mime.text import MIMEText

	sender = 'yourmailaddress@mailbox.com'
	password = 'yourpassword'
	try:
		s=smtplib.SMTP()
		s.connect("smtp.gmail.com",587)
		s.starttls()
		s.login(sender, password)
		msg = MIMEText(Message)
		msg['To'] = email.utils.formataddr(('Recipient', receiver))
		msg['From'] = email.utils.formataddr(('Server', sender))
		msg['Subject'] = 'Coding Finished Message'
		s.sendmail(sender, receiver, msg.as_string())
	except Exception,R:
		return R
	finally:
		s.quit()
