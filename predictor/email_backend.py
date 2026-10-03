import resend

from django.conf import settings
from django.core.mail.backends.base import BaseEmailBackend


class ResendEmailBackend(BaseEmailBackend):

    def send_messages(self, email_messages):
        if not email_messages:
            return 0

        resend.api_key = settings.RESEND_API_KEY
        sent_count = 0

        for message in email_messages:
            params = {
                "from": settings.DEFAULT_FROM_EMAIL,
                "to": list(message.to),
                "subject": message.subject,
                "text": message.body,
            }

            # Use HTML email if Django/allauth provides one
            for content, mimetype in message.alternatives:
                if mimetype == "text/html":
                    params["html"] = content
                    break

            try:
                resend.Emails.send(params)
                sent_count += 1
            except Exception:
                if not self.fail_silently:
                    raise

        return sent_count

