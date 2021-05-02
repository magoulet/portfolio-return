import sys
import smtplib
# from email.MIMEMultipart import MIMEMultipart
# from email.MIMEText import MIMEText
import requests


def sendmail(fromaddr, to, subject, body):

    key = "1f2a796506e8847864ef3a9e9f9a205c-2b0eef4c-1e1f9688"
    sandbox = "sandbox7031160c38b54f3db2765cbe20acdc25.mailgun.org"
    mailer = 'mailgun'

    if mailer == "mailgun":
            # The actual mail sending (mailgun sandboxed url)                                                                      
        request_url =  'https://api.mailgun.net/v2/{0}/messages'.format(sandbox)
        request = requests.post(request_url, auth=('api', key), data={
                  'from': fromaddr,
                  'to': to,
                  'subject': subject,
                  'text': body
                })

    else:
        # The actual mail sending (Gmail SMTP server)                                                                          
        server = smtplib.SMTP('smtp.gmail.com',587)
        server.starttls()
        server.login(username,password)
        server.sendmail(fromaddr, toaddrs, message)
        server.quit()

    return

def main():
    try:
        sendmail(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    except:
        print('Please enter the following variables: \nFrom, To, Subject, Body (Total 4 arguments)')

if __name__ == "__main__":
    main()

