"""The renderers: views from `calm_additive.views` → text or figures.

One module per output — `text` (printed), `mpl` (matplotlib). A renderer
takes a view and draws it; it computes nothing about the model and never
imports the engine, so another look or plotting package is another module
here.
"""
