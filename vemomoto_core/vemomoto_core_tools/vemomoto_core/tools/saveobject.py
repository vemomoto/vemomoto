'''
This module contains methods and classes to save and load objects.

The module contains a method :py:meth:`save_object` that saves any object without 
overwriting existing identical sections. Furthermore, there is a class 
:py:class:`SeparatelySaveable` that can be used as the base class for all objects for 
which some attributes shall be saved in separate files. 
Attributes that are instances of :py:class:`SeparatelySaveable` also will be 
saved separately automatically. Further attributes that shall be saved 
separately can be specified via :py:meth:`SeparatelySaveable.set_save_separately`. 

Attributes that are saved separately are placed in a folder next to the 
file to which the original object is saved. These attributes will be 
saved again only if they have been accessed after they have been saved 
initially. When the object is loaded, the separate attributes will not be 
loaded until they are accessed.

Usage
-----

Saving an object without overwriting similar parts works for all objects:

.. code-block:: python

    save_object(myObject)

Saving attributes separately:

.. code-block:: python

    # defining classes
    class MyClass1(SeparatelySaveable):
        def __init__(self, value):
            super().__init__()
            self.attribute1 = value
            
            # specify that self.attribute shall be 
            # saved separately
            self.set_save_separately('attribute')
        
    class MyClass2(SeparatelySaveable):
        def __init__(self, value):
            super().__init__()
            
            # attributes that are instances of 
            # SeparatelySaveable will always be saved separately
            self.attribute2 = MyClass1(value)
    
    
    # creating objects
    myObject1 = MyClass1()
    myObject2 = MyClass2()
    
    # Saves myObject1 to fileName1.ext and 
    # myObject1.attribute1 to fileName1.arx/attribute1.ext
    myObject1.save_object("fileName1", ".ext", ".arx")
    
    # Saves myObject2 to fileName2.ext and 
    # myObject2.attribute2 to fileName2.arx/attribute2.ext and
    # myObject2.attribute2.attribute1 to fileName2.arx/attribute2.arx/attribute1.ext
    myObject1.save_object("fileName2", ".ext", ".arx")
    
    # load myObject2; myObject2.attribute2 will remain unloaded
    loadedObject = load_object("fileName2.ext")
    
    # myObject2.attribute1 will be loaded; myObject2.attribute2.attribute1 
    # will remain unloaded
    loadedObject.attribute2
    
    # Saves loadedObject to fileName2.ext and 
    # loadedObject.attribute2 to fileName2.arx/attribute2.ext 
    # loadedObject.attribute2.attribute1 will remain untouched
    loadedObject.save_object("fileName2", ".ext", ".arx")

'''

import dill
import os
import io
from itertools import count
from astropy.wcs.docstrings import name

DEFAULT_EXTENSION = ''
"""File name extension used if no extension is specified"""

DEFAULT_FOLDER_EXTENSION = '.arx'
"""Folder name extension used if no extension is specified"""

BLOCKSIZE = 2**20
"""Size of read/write blocks when files are saved"""


def load_object(fileName):
    """Load an object.
    
    Parameters
    ----------
    fileName : str
        Path to the file
    
    """
    with open(fileName, 'rb') as file:
        return dill.load(file)

def save_object(obj, fileName, compare=True):
    """Save an object.
    
    If the object has been saved at the same file earlier, only the parts 
    are overwritten that have changed. Note that an additional attribute 
    at the beginning of the file will 'shift' all data, making it 
    necessary to rewrite the entire file.
    
    Parameters
    ----------
    obj : object
        Object to be saved
    fileName : str
        Path of the file to which the object shall be saved
    compare : bool
        Whether only changed parts shall be overwitten. A value of `True` will
        be beneficial for large files if no/few changes have been made. A 
        value of `False` will be faster for small and strongly changed files.
    
    """
    
    if not compare or not os.path.isfile(fileName):
        with open(fileName, 'wb') as file:
            dill.dump(obj, file, byref=True)
        return
            
    stream = io.BytesIO()
    dill.dump(obj, stream, byref=True)
    stream.seek(0)
    buf_obj = stream.read(BLOCKSIZE)
    with open(fileName, 'rb+') as file:
        buf_file = file.read(BLOCKSIZE)
        for position in count(0, BLOCKSIZE):
            if not len(buf_obj) > 0:
                file.truncate()
                break
            elif not buf_obj == buf_file:
                file.seek(position)
                file.write(buf_obj)
                if not len(buf_file) > 0:
                    file.write(stream.read())
                    break
            buf_file = file.read(BLOCKSIZE)
            buf_obj = stream.read(BLOCKSIZE)
                

class SeparatelySaveable():
    def __init__(self, extension=DEFAULT_EXTENSION, 
                 folderExtension=DEFAULT_FOLDER_EXTENSION):
        self.__dumped_attributes = {}
        self.__archived_attributes = {}
        self.extension = extension
        self.folderExtension = folderExtension
        self.__saveables = set()
    
    def set_save_separately(self, *name):
        self.__saveables.update(name)
    
    def del_save_separately(self, *name):
        self.__saveables.difference_update(name)
    
    def __getattr__(self, name):
        # prevent infinite recursion if object has not been correctly initialized
        if (name == '_SeparatelySaveable__archived_attributes' or 
            name == '_SeparatelySaveable__dumped_attributes'): 
            raise AttributeError('SeparatelySaveable object has not been '
                                 'initialized properly.')
        
        if name in self.__archived_attributes:
            value = self.__archived_attributes.pop(name)
        elif name in self.__dumped_attributes:
            value = load_object(self.__dumped_attributes.pop(name))
        else:
            raise AttributeError("'" + type(self).__name__ + "' object "
                                 "has no attribute '" + name + "'")
        setattr(self, name, value)
        return value
    
    def __delattr__(self, name):
        try:
            self.__dumped_attributes.pop(name)
            try:
                super().__delattr__(name)
            except AttributeError:
                pass
        except KeyError:
            super().__delattr__(name)
    
    def hasattr(self, name):
        if name in self.__dumped_attributes or name in self.__archived_attributes:
            return True
        else:
            return hasattr(self, name)
    
    def load_all(self):
        for name in list(self.__archived_attributes):
            getattr(self, name)
        for name in list(self.__dumped_attributes):
            getattr(self, name)
    
    def save_object(self, fileName, extension=None, folderExtension=None,
                    overwriteChildExtension=False):
        if extension is None:
            extension = self.extension
        if folderExtension is None:
            folderExtension = self.folderExtension
        
        # account for a possible name change - load all components
        # if necessary; this could be done smarter
        if not (self.__dict__.get('_SeparatelySaveable__fileName', 
                                  None) == fileName
                and self.__dict__.get('_SeparatelySaveable__extension', 
                                      None) == extension
                and self.__dict__.get('_SeparatelySaveable__folderExtension', 
                                      None) == folderExtension
                and self.__dict__.get('_SeparatelySaveable__overwriteChildExtension', 
                                      None) == overwriteChildExtension
                ):
            self.__fileName = fileName
            self.__extension = extension
            self.__folderExtension = folderExtension
            self.__overwriteChildExtension = overwriteChildExtension
            self.load_all()
            
            
        # do not save the attributes that had been saved earlier and have not
        # been accessed since
        archived_attributes_tmp = self.__archived_attributes
        self.__archived_attributes = {}
        
        # save the object
        dumped_attributes_tmp = {}
        saveInFolder = False
        for name, obj in self.__dict__.items():
            if isinstance(obj, SeparatelySaveable) or name in self.__saveables:
                if not saveInFolder:
                    folderName = fileName+folderExtension
                    if not os.access(folderName, os.F_OK):
                        os.makedirs(folderName)
                    saveInFolder = True
                partFileName = os.path.join(folderName, name)
                if isinstance(obj, SeparatelySaveable):
                    if overwriteChildExtension:
                        savedFileName = obj.save_object(partFileName, extension, 
                                                        folderExtension,
                                                        overwriteChildExtension)
                    else:
                        savedFileName = obj.save_object(partFileName)
                else:
                    savedFileName = partFileName+extension
                    save_object(obj, savedFileName)
                dumped_attributes_tmp[name] = obj
                self.__dumped_attributes[name] = savedFileName
        
        for name in dumped_attributes_tmp:
            self.__dict__.pop(name)
            
        save_object(self, fileName+extension)
        
        archived_attributes_tmp.update(dumped_attributes_tmp)
        self.__archived_attributes = archived_attributes_tmp
            
        return fileName+extension
        