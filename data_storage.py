
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

employees = [Employee("Ashraf Abdelraouf Fathy Abdelshakour"),
Employee("Wassim Abou-Melhem"),
Employee("Ramesh Adusumilli"),
Employee("Peter Joseph Aernoudts"),
Employee("Lana Al-Salem Saqer"),
Employee("Vasudeva Rao Aourpally"),
Employee("Andre Jorge Asano"),
Employee("Denis Balazuc"),
Employee("Cristhian Balderas"),
Employee("Eliseu Folego Baldo"),
Employee("Amanda Baldwin"),
Employee("Arran Gregory Bartish"),
Employee("Ariella Baston"),
Employee("Yacine Belala"),
Employee("Tatiana Bogatchkina"),
Employee("Sebastien Boissonneault"),
Employee("Ioana Nora Burdujoc"),
Employee("Richard Bussiere"),
Employee("Antonio Francesco Caligiuri"),
Employee("Nicole Gina Calinoiu"),
Employee("Juan Enrique Cardenas-Medina"),
Employee("Cedric Caron"),
Employee("David Caya"),
Employee("Eyad Chikh-ibrahim"),
Employee("Manish Chugh"),
Employee("Richard Comtois"),
Employee("Flavius Costa"),
Employee("Sean Coull"),
Employee("Valerie Demeule"),
Employee("Gabriel Desjardins"),
Employee("Angelo Dipaolo"),
Employee("Dang Khoa Do"),
Employee("Karine Dupont"),
Employee("Luis Alberto Estefan Elola"),
Employee("Serra Erkal"),
Employee("Georges Fadous"),
Employee("Laurent Fernandez"),
Employee("Diana Firdjanova"),
Employee("Samantha Fornari"),
Employee("John Gammon"),
Employee("Assen Garbev"),
Employee("Martin Godin"),
Employee("Kasivisweswara Sharma Gollapudi"),
Employee("Florian Gombert"),
Employee("Ke Gong"),
Employee("Atheer Hanna"),
Employee("Maddison Harder"),
Employee("Gabriel Hernandez"),
Employee("Erin Hollen"),
Employee("Tenveer Hussain"),
Employee("Raynold JR Jean"),
Employee("David W Johnson"),
Employee("Benoit Joly"),
Employee("Thomas Joubin"),
Employee("Behnam Karimi"),
Employee("Bruce M Kearney"),
Employee("Mohamed Kleit"),
Employee("Vyacheslav Kostin"),
Employee("Sergey Kovalenko"),
Employee("Stephanie Kulovics"),
Employee("Arvind Kumar"),
Employee("Eric Langlais"),
Employee("Jean-Philippe LeBlanc"),
Employee("Lindy Li Loong"),
Employee("Julie Anne Lichocki"),
Employee("Quan Liu"),
Employee("Anne Lizotte"),
Employee("Amarbir Singh Lubana"),
Employee("Cherifa Mansoura Liamani"),
Employee("Paul Anthony Moore"),
Employee("Olivier Mornet"),
Employee("Deyvisson Oliveira"),
Employee("Erwin Pant"),
Employee("Jaswanth Paruchuri"),
Employee("Liza Perreault"),
Employee("Oksana  Poliarush"),
Employee("Leonardo Postacchini"),
Employee("Aneesh Potluri"),
Employee("Maria-Luisa Quintana"),
Employee("Seddik Ramdani"),
Employee("Muhammad Ammar"),
Employee("Bruno Luiz S Ribeiro"),
Employee("Nicolas Richard"),
Employee("Sergio Romero Del Bosque"),
Employee("Sergei Rybakov"),
Employee("Venkata R Sathi"),
Employee("Ralf Schneider"),
Employee("Sasitharan Sellathurai"),
Employee("Vithiyatharan Sellathurai"),
Employee("Yury Shamne"),
Employee("Abhishek Sharma"),
Employee("Nidhi Sharma"),
Employee("Dzvenyslava Siatetska"),
Employee("Roxanne Sirois"),
Employee("Daniel Tardif"),
Employee("Vincent-Julien Tortajada"),
Employee("Erik Trepanier"),
Employee("Pawel Adam Urban"),
Employee("Vannadine Ven"),
Employee("Artem Volynets"),
Employee("Chirag Vyas"),
Employee("Vaishalibahen Vyas"),
Employee("Ruo Nong Wang"),
Employee("Zhihua Xi"),
Employee("Jie Xu"),
Employee("Larry Belanger"),
Employee("Ron Andrew Callender"),
Employee("Leah Cerro"),
Employee("Jason Cote"),
Employee("Beverly Darlene Dentry"),
Employee("Feigang Fei"),
Employee("Patrick Guindon-Slater"),
Employee("Maria Eleftheria Hatajlo"),
Employee("Jean-Francois Lemire"),
Employee("Steven McGurn"),
Employee("Jeannie Riel"),
Employee("Jean Nicholas Thomas"),
Employee("Oleg Volkovich")
]

def get_employee_names_list():
    employee_names = []
    for employee in employees:
        employee_names.append(employee.get_name())
    return employee_names