
class Snack:

    def __init__(self, name, price):
        self.name = name
        self.price = price

    def get_price(self):
        return self.price

    def get_name(self):
        return self.name


snacks = [Snack("chocolate", 1), Snack("chips", 0.75), Snack("drink", 0.5)]


class Employee:

    def __init__(self, name):
        self.name = name
        self.total = 0

    def add_to_total(self, amount):
        self.total = self.total + amount

    def get_total(self):
        return self.total

    def get_name(self):
        return self.name

employees = [Employee("Abdelshakour,Ashraf Abdelraouf Fathy"),
Employee("Abou-Melhem,Wassim"),
Employee("Adusumilli,Ramesh"),
Employee("Aernoudts,Peter Joseph"),
Employee("Al-Salem Saqer,Lana"),
Employee("Aourpally,Vasudeva Rao"),
Employee("Asano,Andre Jorge"),
Employee("Balazuc,Denis"),
Employee("Balderas,Cristhian"),
Employee("Baldo,Eliseu Folego"),
Employee("Baldwin,Amanda"),
Employee("Bartish,Arran Gregory"),
Employee("Baston,Ariella"),
Employee("Belala,Yacine"),
Employee("Bogatchkina,Tatiana"),
Employee("Boissonneault,Sebastien"),
Employee("Burdujoc,Ioana Nora"),
Employee("Bussiere,Richard"),
Employee("Caligiuri,Antonio Francesco"),
Employee("Calinoiu,Nicole Gina"),
Employee("Cardenas-Medina,Juan Enrique"),
Employee("Caron,Cedric"),
Employee("Caya,David"),
Employee("Chikh-ibrahim,Eyad"),
Employee("Chugh,Manish"),
Employee("Comtois,Richard"),
Employee("Costa,Flavius"),
Employee("Coull,Sean"),
Employee("Demeule,Valerie"),
Employee("Desjardins,Gabriel"),
Employee("Dipaolo,Angelo"),
Employee("Do,Dang Khoa"),
Employee("Dupont,Karine"),
Employee("Estefan Elola,Luis Alberto"),
Employee("Erkal,Serra"),
Employee("Fadous,Georges"),
Employee("Fernandez,Laurent"),
Employee("Firdjanova,Diana"),
Employee("Fornari,Samantha"),
Employee("Gammon,John"),
Employee("Garbev,Assen"),
Employee("Godin,Martin"),
Employee("Gollapudi,Kasivisweswara Sharma"),
Employee("Gombert,Florian"),
Employee("Gong,Ke"),
Employee("Hanna,Atheer"),
Employee("Harder,Maddison"),
Employee("Hernandez,Gabriel"),
Employee("Hollen,Erin"),
Employee("Hussain,Tenveer"),
Employee("Jean,Raynold JR"),
Employee("Johnson,David W"),
Employee("Joly,Benoit"),
Employee("Joubin, Thomas"),
Employee("Karimi,Behnam"),
Employee("Kearney,Bruce M"),
Employee("Mohamed Kleit"),
Employee("Kostin,Vyacheslav"),
Employee("Kovalenko,Sergey"),
Employee("Kulovics,Stephanie"),
Employee("Kumar,Arvind"),
Employee("Langlais, Eric"),
Employee("LeBlanc,Jean-Philippe"),
Employee("Li Loong,Lindy"),
Employee("Lichocki, Julie Anne"),
Employee("Liu,Quan"),
Employee("Lizotte,Anne"),
Employee("Lubana,Amarbir Singh"),
Employee("Mansoura Liamani,Cherifa"),
Employee("Moore, Paul Anthony"),
Employee("Mornet,Olivier"),
Employee("Oliveira,Deyvisson"),
Employee("Pant,Erwin"),
Employee("Paruchuri,Jaswanth"),
Employee("Perreault,Liza"),
Employee("Poliarush,Oksana "),
Employee("Postacchini,Leonardo"),
Employee("Potluri,Aneesh"),
Employee("Quintana,Maria-Luisa"),
Employee("Ramdani,Seddik"),
Employee("Ammar, Muhammad"),
Employee("Ribeiro,Bruno Luiz S"),
Employee("Richard,Nicolas"),
Employee("Romero Del Bosque,Sergio"),
Employee("Rybakov,Sergei"),
Employee("Sathi,Venkata R"),
Employee("Schneider,Ralf"),
Employee("Sellathurai,Sasitharan"),
Employee("Sellathurai,Vithiyatharan"),
Employee("Shamne,Yury"),
Employee("Sharma,Abhishek"),
Employee("Sharma,Nidhi"),
Employee("Siatetska,Dzvenyslava"),
Employee("Sirois,Roxanne"),
Employee("Tardif,Daniel"),
Employee("Tortajada,Vincent-Julien"),
Employee("Trepanier,Erik"),
Employee("Urban,Pawel Adam"),
Employee("Ven,Vannadine"),
Employee("Volynets,Artem"),
Employee("Vyas,Chirag"),
Employee("Vyas,Vaishalibahen"),
Employee("Wang,Ruo Nong"),
Employee("Xi,Zhihua"),
Employee("Xu,Jie"),
Employee("Belanger,Larry"),
Employee("Callender,Ron Andrew"),
Employee("Cerro,Leah"),
Employee("Cote,Jason"),
Employee("Dentry,Beverly Darlene"),
Employee("Fei,Feigang"),
Employee("Guindon-Slater,Patrick"),
Employee("Hatajlo,Maria Eleftheria"),
Employee("Lemire,Jean-Francois"),
Employee("McGurn,Steven"),
Employee("Riel,Jeannie"),
Employee("Thomas, Jean Nicholas"),
Employee("Volkovich,Oleg")]

def get_employee_names_list():
    employee_names = []
    for employee in employees:
        employee_names.append(employee.get_name())
    return employee_names