from django.db import migrations


def update_site_domain(apps, schema_editor):
    Site = apps.get_model("sites", "Site")

    site, created = Site.objects.get_or_create(
        id=1,
        defaults={
            "domain": "brain-tumor-prediction-w7e0.onrender.com",
            "name": "BrainGAN-SRNet",
        },
    )

    if not created:
        site.domain = "brain-tumor-prediction-w7e0.onrender.com"
        site.name = "BrainGAN-SRNet"
        site.save()


def reverse_site_domain(apps, schema_editor):
    Site = apps.get_model("sites", "Site")

    Site.objects.filter(id=1).update(
        domain="example.com",
        name="example.com",
    )


class Migration(migrations.Migration):

    dependencies = [
        ("predictor", "0004_prediction_scorecam_image"),
        ("sites", "0002_alter_domain_unique"),
    ]

    operations = [
        migrations.RunPython(
            update_site_domain,
            reverse_site_domain,
        ),
    ]

