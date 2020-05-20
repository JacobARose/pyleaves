'''
Script for defining class structures for database objects, for the purpose of standardization, validation, and explicit documentation.
'''

#TODO Add automatic logic for looking up family & genus when provided a species


class TaxonomyMapper:
    
    def __init__(self, map_file):
        '''
        #TODO
        Class for extracting family from genus, and genus from species. Useful for automatically updating higher hierarchy levels (e.g. family) when changing a lower level (e.g. species) for a known sample.
        '''
        
    def get_genus(self, species):
        if type(species) != str:
            print('Must provide species label as str')
            return None
        else:
            #TODO Figure out the taxonomy mapping file
            raise NotImplemented
        
    def get_family(self, species=None, genus=None):
        if genus is None:
            genus = self.get_genus(species)
            
        #TODO Implement logic for getting family from genus
        raise NotImplemented
        


class FamilyItem:
    
    def __init__(self, family=None):
        
        self.set_family(family)
        
    def set_family(self, family=None):
        self.family = family
        
    def get_family(self):
        return self.family

class GenusItem(FamilyItem):
    
    def __init__(self, genus=None, family=None):
        if family:
            super().__init__(family=family)
        self.set_genus(genus)
        
    def set_genus(self, genus=None):
        #TODO Update family value within method
        self.genus = genus
        
    def get_genus(self):
        return self.genus
    
class SpeciesItem(GenusItem):
    
    def __init__(self, species=None, genus=None, family=None):
        if family:
            super().__init__(family=family)
        if genus:
            super().__init__(genus=genus, family=family)
        self.set_genus(genus)
        
    def set_genus(self, genus=None):
        #TODO Update family value within method
        self.genus = genus
        
    def get_genus(self):
        return self.genus