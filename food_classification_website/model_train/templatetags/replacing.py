from django import template

register = template.Library()

@register.filter
def replacing(value): #Custom tag use for replacing '_' with blank to format our class names to visualize better
    return value.replace("_"," ")