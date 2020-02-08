'''
Created on 04.07.2016

@author: Samuel
'''

class HierarchichalPrinter(object):
    '''
    classdocs
    '''


    def __init__(self, parentPrinter=None, inheritFromParent=True, 
                 silent=False):
        '''
        Constructor
        '''
        self.set_parent_printer(parentPrinter)
        inherit = parentPrinter is not None and inheritFromParent
        self.__inheritPrintLevel = inherit
        self.__inheritSilentStatus = inherit
        self.__silentStatus = silent
        self.__printLevel = 0
    
    def set_parent_printer(self, parentPrinter):
        if parentPrinter is not None:
            if not isinstance(parentPrinter, HierarchichalPrinter):
                raise ValueError("The parentPrinter must be of the type "
                                 + "HierarchicalPrinter.")
        else:
            self.__inheritPrintLevel = False
            self.__inheritSilentStatus = False
        self.__parentPrinter = parentPrinter
        
    def set_silent_status(self, silent=None):
        if silent is not None:
            self.__silentStatus = silent
        self.__inheritSilentStatus = False
    
    def set_print_level(self, printLevel=None, absolute=False):
        if printLevel is not None:
            self.__printLevel = printLevel
        if absolute: 
            self.__inheritPrintLevel = False
    
    def increase_print_level(self):
        self.__printLevel += 1
        
    def decrease_print_level(self):
        self.__printLevel -= 1
    
    def inherit_silent_status(self):
        if self.__parentPrinter is None:
            raise Exception("I can only inherit the silent status "
                            + "if a parent is specified. "
                            + "Use set_parent_printer")
        self.__inheritSilentStatus = True
    
    def inherit_print_level(self):
        if self.__parentPrinter is None:
            raise Exception("I can only inherit the print level "
                            + "if a parent is specified. "
                            + "Use set_parent_printer")
        self.__inheritPrintLevel = True
    
    def get_silent_status(self):
        if self.__inheritSilentStatus:
            return self.__parentPrinter.get_silent_status()
        else: 
            return self.__silentStatus
    
    def get_print_level(self):
        if self.__inheritSilentStatus:
            result = self.__parentPrinter.get_print_level() + self.__printLevel
        else: 
            result = self.__printLevel 
        return max(0, result)
    
    def get_parent_printer(self):
        return self.__parentPrinter
    
    def print_status(self, *text, percent=False, noIndent=False, end="\n"):
        
        silent = self.get_silent_status()
        printLevel = self.get_print_level()
        
        if not silent:
            if printLevel and not noIndent:
                print(printLevel * "... ", end="")
            if percent:
                print("{:6.2%}".format(text[0]), end = " ")
                if len(text) > 1:
                    print(*text[1:])
                else:
                    print(end=end)
            else:
                print(*text, end=end)
    
    prst = print_status
                
                
if __name__ == "__main__":
    p1 = HierarchichalPrinter()
    p2 = HierarchichalPrinter(p1)
    p1.increase_print_level()
    p1.prst(1, 2)
    p2.prst(3, 4)
    p2.increase_print_level()
    p2.prst(5, 6)
    p1.decrease_print_level()
    p2.prst(7, 8)
    p1.set_silent_status(True)
    p1.prst("nothing")
    p2.prst("nothing")
    p2.set_silent_status(False)
    p2.prst(9, 10)
    