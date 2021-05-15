from django.contrib import admin
from import_export.admin import ImportExportModelAdmin
from .models import Post
from .models import visuals
# Register your models here.

admin.site.register(visuals)

@admin.register(Post)
class PersonAdmin(ImportExportModelAdmin):
    pass
