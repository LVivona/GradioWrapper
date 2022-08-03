# gradioWrapper 🎁

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/LVivona/GradioWrapper/blob/main/LICENSE)

``@Author Luca Vivona 🙈``

``Github`` [github/LVivona](https://github.com/LVivona/gradio_wrap)
## Table of contents 
- v0.0.7


  - [Quick Start Import](#quick-start-import)


  - [Whats New in v0.0.7](#whats-new-in-v007)


  - [What is it?](#what-is-it-)


  - [How does it work?](#how-does-it-work-)


  - [Class functional decorator](#class-functional-decorator)


  - [Class decorator](#class-decorator)


  - [Examples](#examples-)


  - [How To Run](#how-to-run-%EF%B8%8F)


### Quick Start Import
```python
from gradioWrapper import register, GradioCompiler, functionalCompiler, tabularGradio
```

## Whats New in v0.0.7
- Genral Functional Decorator
- Genral Tablular Functional Decorator


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
def register(inputs, outputs, examples=None):
    def register_gradio(func):
        def wrap(self, *args, **kwargs):            
            try:
                self.registered_gradio_functons
            except AttributeError:
                print("✨Initializing Class Functions...✨\n")
                self.registered_gradio_functons = dict()

            fn_name = func.__name__ 
            if fn_name in self.registered_gradio_functons: 
                result = func(self, *args, **kwargs)
                return result
            else:
                self.registered_gradio_functons[fn_name] = dict(inputs=inputs, outputs=outputs, examples=examples)
                return None
        return wrap
    return register_gradio
```

### Class Decorator

```py
def GradioCompiler(cls):
    class GradioWrapper:

        def __init__(self) -> None:
            self.cls = cls()


        def get_funcs(self):
            return [func for func in dir(self.cls) if not func.startswith("__") and type(getattr(self.cls, func, None)) == type(self.get_funcs) ]

        def compile(self, **kwargs):
            print("Just putting on the finishing touches... 🔧🧰")
            for func in self.get_funcs():
                this = getattr(self.cls, func, None)
                if this.__name__ == "wrap":
                    this()

            demos, names = [], []
            for func, param in self.get_registered_gradio_functons().items():                
                names.append(func)
                demos.append(gr.Interface(fn=getattr(self.cls, func, None),
                                            inputs=param['inputs'],
                                            outputs=param['outputs'],
                                            examples=param['examples'],
                                            cache_examples=kwargs['cache_examples'] if "cache_examples" in kwargs else None,
                                            examples_per_page=kwargs['cache_examples'] if "cache_examples" in kwargs else 10,
                                            interpretation=kwargs['interpretation'] if "interpretation" in kwargs else None,
                                            num_shap=kwargs['num_shap'] if "num_shap" in kwargs else 2.0,
                                            title=kwargs['title'] if "title" in kwargs else None,
                                            article=kwargs['article'] if "article" in kwargs else None,
                                            thumbnail=kwargs['thumbnail'] if "thumbnail" in kwargs else None,
                                            css=kwargs['css'] if "css" in kwargs else None,
                                            live=kwargs['live'] if "live" in kwargs else False,
                                            allow_flagging=kwargs['allow_flagging'] if "allow_flagging" in kwargs else None,
                                            theme='default', 
                                            ))
                print(f"{func}....{bcolor.BOLD}{bcolor.OKGREEN} done {bcolor.ENDC}")

            print("\nHappy Visualizing... 🚀")
            return gr.TabbedInterface(demos, names)
            
        def get_registered_gradio_functons(self):
            try:
                self.cls.registered_gradio_functons
            except AttributeError:
                return None
            return self.cls.registered_gradio_functons
        

        def run(self, **kwargs):
            port= kwargs["port"] if "port" in kwargs else DOCKER_PORT.determinePort() 

            self.compile(live=kwargs[ 'live' ] if "live" in kwargs else False,
                         allow_flagging=kwargs[ 'allow_flagging' ] if "allow_flagging" in kwargs else None,
                         cache_examples=kwargs['cache_examples'] if "cache_examples" in kwargs else None,
                         examples_per_page=kwargs['cache_examples'] if "cache_examples" in kwargs else 10,
                         interpretation=kwargs['interpretation'] if "interpretation" in kwargs else None,
                         num_shap=kwargs['num_shap'] if "num_shap" in kwargs else 2.0,
                         title=kwargs['title'] if "title" in kwargs else None,
                         article=kwargs['article'] if "article" in kwargs else None,
                         thumbnail=kwargs['thumbnail'] if "thumbnail" in kwargs else None,
                         css=kwargs['css'] if "css" in kwargs else None,
                         theme=kwargs['theme'] if "theme" in kwargs else None, 
                         ).launch(server_port=port,
                                  inline= kwargs['inline'] if "inline" in kwargs else True,
                                  share=kwargs['share'] if "share" in kwargs else None,
                                  debug=kwargs['debug'] if "debug" in kwargs else False,
                                  enable_queue=kwargs['enable_queue'] if "enable_queue" in kwargs else None,
                                  max_threads=kwargs['max_threads'] if "max_threads" in kwargs else None,
                                  auth=kwargs['auth'] if "auth" in kwargs else None,
                                  auth_message=kwargs['auth_message'] if "auth_message" in kwargs else None,
                                  prevent_thread_lock=kwargs['prevent_thread_lock'] if "prevent_thread_lock" in kwargs else False,
                                  show_error=kwargs['show_error'] if "show_error" in kwargs else True,
                                  show_tips=kwargs['show_tips'] if "show_tips" in kwargs else False,
                                  height=kwargs['height'] if "height" in kwargs else 500,
                                  width=kwargs['width'] if "width" in kwargs else 900,
                                  encrypt=kwargs['encrypt'] if "encrypt" in kwargs else False,
                                  favicon_path=kwargs['favicon_path'] if "favicon_path" in kwargs else None,
                                  ssl_keyfile=kwargs['ssl_keyfile'] if "ssl_keyfile" in kwargs else None,
                                  ssl_certfile=kwargs['ssl_certfile'] if "ssl_certfile" in kwargs else None,
                                  ssl_keyfile_password=kwargs['ssl_keyfile_password'] if "ssl_keyfile_password" in kwargs else None,
                                  quiet=kwargs['quiet'] if "quiet" in kwargs else False) 


    return GradioWrapper
```

### Functional Decorator
```python
def functionalCompiler(inputs, outputs, **kwargs):
    def register_func(func):
        def wrap():
            inter = gr.Interface(fn=func,
                                 inputs=inputs,
                                 outputs=outputs,
                                 examples=kwargs['examples'] if "examples" in kwargs else None,
                                 live=kwargs[ 'live' ] if "live" in kwargs else False,
                                 allow_flagging=kwargs[ 'allow_flagging' ] if "allow_flagging" in kwargs else None,
                                 cache_examples=kwargs['cache_examples'] if "cache_examples" in kwargs else None,
                                 examples_per_page=kwargs['cache_examples'] if "cache_examples" in kwargs else 10,
                                 interpretation=kwargs['interpretation'] if "interpretation" in kwargs else None,
                                 num_shap=kwargs['num_shap'] if "num_shap" in kwargs else 2.0,
                                 title=kwargs['title'] if "title" in kwargs else None,
                                 article=kwargs['article'] if "article" in kwargs else None,
                                 thumbnail=kwargs['thumbnail'] if "thumbnail" in kwargs else None,
                                 css=kwargs['css'] if "css" in kwargs else None,
                                 theme=kwargs['theme'] if "theme" in kwargs else None)
            return inter
        return wrap
    return register_func
```
## Examples 🧪

#### Functional Example
```python
@functionalCompiler(inputs=[gr.Textbox(label="name")], outputs=['text'])
def Hello_World(name):
        return f"Hello {name}, and welcome to Gradio Flow 🤗" 

@functionalCompiler(inputs=[gr.Textbox(label="name")], outputs=['text'])
def Goodbye(name):
        return f"Goodbye {name}" 

```


#### Class Example
```python
@GradioCompiler
class Greeting:

    @register(inputs=[gr.Textbox(label="name")], outputs=['text'])
    def Hello_World(self, name):
        return f"Hello {name}, and welcome to Gradio Flow 🤗" 

    @register(inputs=[gr.Textbox(label="name")], outputs=['text'])
    def Goodbye(name):
        return f"Goodbye {name}" 

```
## How To Run ⚙️

#### Run Class
```python

# Greeting class from Class Example
###################################
a = Greeting()
a.run() # or a.run()

```

#### Run Singular Function
```python

# HelloWorld function from Function Example
###################################
HelloWorld().launch()

```

### Run Multiple Functions
```python

# HelloWorld, and Goodbye function from Function Example
###################################
tabularGradio([Hello_World(),Goodbye()],["Hello World", "Goodbye"])

```
