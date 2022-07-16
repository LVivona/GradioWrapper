# gradio_wrapper 🎁
``@Author Luca Vivona 🙈``

## Table of contents 
- v0.01
  - Quick Start Import
  - What is it?
  - How does it work?
  - Class function decorator
  - Class decorator
  - Examples
  - v0.01 compared to v0.00
  - How To Run

### **New** v0.01 🥳🎉

### Quick Start Import
```python
from gradio_wrapper import register, gradio_compile
```

## What is it? 🤨
In essence it's extension to the gradio, by using wrappers/decorators that is built into python I'm able wrap class function into overlapping function and compile the information required to run local gradio applications 

## How does it work? 🤔
There are two major things that allow this to be possible in which will go into, and those are function wrapper, and the class wrapper.

**What are wrapper/decorators?** Decorators/wrappers is a powerful tool in python represented by a @ and position like in the example below
```python
@nameofdecorator 
def Foo(*args, **kwargs):
    ...
```

that allow us to wrap another function in order to extend the behavior of the wrapped function, without permanently modifying it. In Decorators, functions are taken as the argument into another function and then called inside the wrapper function. How this tool is essential used is like gradio a Function has an input and output and as the user were able to define which function we want to convert into a UI. By using the decorators function the wrapper is able to store that information within a dictionary with the key representing the name of the function and the values being an dictionary of input and output holding the information you would put into your 

```python 
Import gradio as gr
gr.interface(fn=Foo, inputs=[...], outputs=[...], ...)
```
inputs and outputs. From there the wrapper class initializer will call the compile function which would read the dictionary and put it in a tabular Interface which gradio provides.

### Class Functional decorator

```python
import gradio as gr
from src.ports import determinePort

def register(inputs, outputs):
    def register_gradio(func):
        def wrap(self, *args, **kwargs):

            try:
                self.registered_gradio_functons
            except AttributeError:
                print(":sparkles:Initializing Functions...:sparkles:")
                self.registered_gradio_functons = dict()

            fn_name = func.__name__ 
            if fn_name in self.registered_gradio_functons: 
                result = func(*args, **kwargs)
                return result
            else:
                self.registered_gradio_functons[fn_name] = dict(inputs=inputs, outputs=outputs)
                return None
        return wrap
    return register_gradio
    
```

### Class Decorator

```py
def gradio_compile(cls):

    class Wrapper:
        def __init__(self) -> None:
            self.cls = cls()

        def get_func(self):
            return [func for func in dir(self.cls) if not func.startswith("__") and type(getattr(self.cls, func, None)) == type(self.get_func) ]
    
        def _compile(self, **kwargs):
            print("Just putting on the finishing touches... :wrench::toolbox:")
            for func in self.get_func():
                this = getattr(self.cls, func, None)
                if this.__name__ == "wrap":
                    this()

            demos = []
            names= []
            for func, param in self.get_registered_gradio_functons().items():
                print(param)
                
                demos.append(gr.Interface(fn=getattr(self.cls, func, None),
                                          inputs=param['inputs'],
                                          outputs=param['outputs'],
                                          live=kwargs['live'] if "live" in kwargs else False,
                                          allow_flagging=kwargs['flagging'] if "flagging" in kwargs else 'never',
                                          share=kwargs['share'] if "share" in kwargs else False,
                                          theme='default'))
                names.append(func)
            print("Happy Visualizing... :rocket:")
            return gr.TabbedInterface(demos, names)
            
        def get_registered_gradio_functons(self):
            try:
                self.cls.registered_gradio_functons
            except AttributeError:
                return None
            return self.cls.registered_gradio_functons
            
        def run(self, **kwargs):
                port= kwargs["port"] if "port" in kwargs else determinePort() 
                return self.compile(live=kwargs['live'] if "live" in kwargs else False,
                                    allow_flagging=kwargs['flagging'] if "flagging" in kwargs else 'never',
                                    share=kwargs['share'] if "share" in kwargs else False).launch() 
    
    return Wrapper
```

## Examples 🧪

```python
from gradio_wrapper import register, gradio_compile
import gradio as gr
#...
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from torch import nn
from pathlib import Path

@gradio_compile
class Pictionary:

    def __init__(self) -> None:
        self.LABELS = Path('./src/examples/data/labels.txt').read_text().splitlines()
    
        self.model = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(1152, 256),
                nn.ReLU(),
                nn.Linear(256, len(self.LABELS)),
                )   
        state_dict = torch.load('./src/examples/data/pytorch_model.bin',    map_location='cpu')
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    @register(inputs="sketchpad", outputs=gr.Label())
    def perdict(self, img) -> 'dict[str, float]':
        if type(img) == type(None): return {}
        x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.
        with torch.no_grad():
            out = self.model(x)
        probabilities = torch.nn.functional.softmax(out[0], dim=0)
        values, indices = torch.topk(probabilities, 5)
        confidences = {self.LABELS[i]: v.item() for i, v in zip(indices, values)}
        return confidences

@gradio_compile
class HelloWorld_2_0:


    @register(inputs=["text", "text", gr.Radio(["morning", "evening", "night"])], outputs="text")
    def Hello(self, Lname : str, Fname : str, day : 'list[any]'=["morning", "evening", "night"]) -> str:
        return "Hello, {} {}".format(Fname, Lname)  

    @register(inputs=["text", "text"], outputs="text")
    def goodbye(self, Fname : str, Lname : str) -> str:
        return "Goodbye, {} {}".format(Fname, Lname)  
    
    @register(inputs=["text", gr.Checkbox() , gr.Slider(0, 60)], outputs=["text", "number"])
    def greet(self, name, is_morning, temperature):
        salutation = "Good morning" if is_morning else "Good evening"
        greeting = "%s %s. It is %s degrees today" % (salutation, name, temperature)
        celsius = (temperature - 32) * 5 / 9
        return (greeting, round(celsius, 2))


@gradio_compile
class FSD:

    def get_new_val(self, old_val, nc):
        return np.round(old_val * (nc - 1)) / (nc - 1)


    def palette_reduce(self, img : PIL.Image.Image, nc : 'tuple[float, float, float]'=(0.0000, 0, 16)):
        pixels = np.array(img, dtype=float) / 255
        pixels = self.get_new_val(pixels, nc)

        carr = np.array(pixels / np.max(pixels) * 255, dtype=np.uint8)
        return PIL.Image.fromarray(carr)

    @register(inputs=[gr.Image(), gr.Slider(0.00, 16)], outputs=gr.Gallery())
    def Floyd_Steinberg_dithering(self, img : PIL.Image.Image="pill", nc : 'tuple[float, float, float]'=(0.0000, 0, 16) ) -> 'list[PIL.Image.Image]':
        pixels = np.array(img, dtype=float) / 255
        new_height, new_width, _ = img.shape 
        for row in range(new_height):
            for col in range(new_width):
                old_val = pixels[row, col].copy()
                new_val = self._get_new_val(old_val, nc)
                pixels[row, col] = new_val
                err = old_val - new_val
                if col < new_width - 1:
                    pixels[row, col + 1] += err * 7 / 16
                if row < new_height - 1:
                    if col > 0:
                        pixels[row + 1, col - 1] += err * 3 / 16
                    pixels[row + 1, col] += err * 5 / 16
                    if col < new_width - 1:
                        pixels[row + 1, col + 1] += err * 1 / 16
        carr = np.array(pixels / np.max(pixels, axis=(0, 1)) * 255, dtype=np.uint8)
        return [PIL.Image.fromarray(carr), self.palette_reduce(img, nc) ]



@gradio_compile
class C:

    def Hello(self):
        return "Hello"
    
    @register(inputs="text", outputs="text")
    def Greeting(self, name):
        return self.Hello() + " " + name

@gradio_compile
class stock_forecast:
    
    def __init__(self):
        matplotlib.use('Agg')

    @register(inputs=[gr.Checkbox(label="legend"), gr.Radio([2025, 2030, 2035, 2040], label="projct"), gr.CheckboxGroup(["Google", "Microsoft", "Gradio"], label="company"), gr.Slider(label="noise"), gr.Radio(["cross", "line", "circle"], label="style")], outputs=[gr.Plot()])
    def plot_forcast(self, legend, project, companies , noise , styles)-> matplotlib.figure.Figure:
        start_year = 2022
        x = np.arange(start_year, project + 1)
        year_count = x.shape[0]
        plt_format = ({"cross": "X", "line": "-", "circle": "o--"})[styles]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, company in enumerate(companies):
            series = np.arange(0, year_count, dtype=float)
            series = series**2 * (i + 1)
            series += np.random.rand(year_count) * noise
            ax.plot(x, series, plt_format)
        if legend:
            plt.legend(companies)
        print(type(fig))
        return fig 

```

## How to Run ⚙️
After you add the decorators on the function you want you can initialize the class and call the function run(**kwargs). In the example below I will call a class from one of my examples section and run the code

```python

a = Pictionary()
a.run(live=True) # or a.run()

```