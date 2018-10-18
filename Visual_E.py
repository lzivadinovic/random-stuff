"""
ver np.pi

Kod za interaktivnu vizualizaciju vektorkog polja oko tačkastih naelektrisanja.

Program koristi sledeće module:
    1. numpy - Osnovne numeričke operacije poput sqrt, abs i funkcije meshgrid i plus su backendovane u nekom nizem programskom jeziku.
    2. scipy.const - Zato što me je mrzelo da nadjem na wiki-ju koliko iznosi dielektrična konstanta vakuuma i da prepišem taj broj.
    3. matplotlib - Vizualizacija svega i svačega. Jako moćan modul ako se pročita dokumentacija. Otkrio sam da postoji opcija da se prave animacije i live prikazi raznih stvari. (recimo talasno kretanje) Potrebno je instalirati i tk ako se ovo instlaira u virtualenv-u

Odlučio sam da ne pišem veliki docstring već da većinu linija komentarišem uz objašnjenje šta rade.

Želim da se zahvalim Dušanu Vukadinović na korisnim savetima i na ispravljanju bagova prilikom pisanja ovog koda.

Lazar Živadinović
Jun 2015.

"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.constants as const

skr = const.pi*const.epsilon_0*4

class DraggablePoint:
    def __init__(self, p, q0, ind): #inicijalizujem telo
        self.q = q0 #sopstveno naelektrisanj je prosledjeno
        self.point = p
        self.c_kruznice = p.center
        self.press = None #definisem varijablu koja prikuplja poziciju nakon klika
        self.indeks = ind #indeks tela

        """definisem funckicju koja mi povezuje dogadjaj u plotu sa mouse click-om i kaze koja funkcija da se izvrsi nakon odredjene akcije
        u ovom slucaju imamo tri pocetak klika dok drzimo i kraj
        """

    def connect(self):
        self.cidpress = self.point.figure.canvas.mpl_connect('button_press_event', self.button_press_event)
        self.cidrelease = self.point.figure.canvas.mpl_connect('button_release_event',self.button_release_event)
        self.cidmotion = self.point.figure.canvas.mpl_connect('motion_notify_event', self.motion_notify_event)

    def disconnect(self): #destructor za vezu
        self.point.figure.canvas.mpl_disconnect(self.cidpress)
        self.point.figure.canvas.mpl_disconnect(self.cidrelease)
        self.point.figure.canvas.mpl_disconnect(self.cidmotion)


        """"na pocetak klika proveravamo da li je klik na grafiku, ako jeste
        menjamo self.press, radi samo za levi klik
        """
    def button_press_event(self,event):
        if event.inaxes != self.point.axes:
            return
        contains = self.point.contains(event)[0]
        if not contains: return
        if event.button == 1:
            self.press = self.point.center, event.xdata, event.ydata


        """Na pomeraj crtamo kruzic tako da uzimamo staru lokaciju i pomeramo za dx/dy
        """
    def motion_notify_event(self, event):
        if self.press is None: return
        if event.inaxes != self.point.axes: return
        self.point.center, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.point.center = (self.point.center[0]+dx, self.point.center[1]+dy)
        self.point.figure.canvas.draw()

        """kada pustimo ocistimo celo polje, nacrtamo kruzice, i opet izracunamo polje sa novim pozicijama
        """
    def button_release_event(self,event):
        self.press = None
        self.point.figure.canvas.draw()
        self.c_kruznice = self.point.center[0], self.point.center[1]
        if self.indeks == 0:
            racun = polje(drs, X, Y)
            ax.cla()
            for c in circles:
                ax.add_patch(c)
            ax.quiver(X,Y,racun[0],racun[1], color='r',  alpha=0.5)
            ax.quiver(X,Y,racun[0],racun[1], edgecolor='k', facecolor='None', linewidth =.5)
            plt.draw()


if __name__ == '__main__':
    f_s=0.3 #force_softening



    def R(x,y):
        r=np.sqrt(x**2+y**2)+f_s
        return r


    def polje(tela, X, Y): #za racunanje polja
        Ex = 0
        Ey = 0
        for i in range(len(tela)):
            r = R(tela[i].c_kruznice[0] - X, tela[i].c_kruznice[1] - Y )
            ex = X - tela[i].c_kruznice[0]
            ey = Y - tela[i].c_kruznice[1]
            Ex += (tela[i].q/skr)*(1/(r+f_s)**3)*ex
            Ey += (tela[i].q/skr)*(1/(r+f_s)**3)*ey
        return Ex, Ey


    fig = plt.figure(figsize=(8,8)) #definisemo velicinu figure
    ax = fig.add_axes([0.05,0.05,0.92,0.92]) #definisemo ose
    ax.set_xlim(-6,6) #podesavamo granice prostora
    ax.set_ylim(-6,6)
    scale = 0.2
    X,Y=np.mgrid[-5:5:scale, -5:5:scale] #definisemo prostor

    circles = []
    q=[-1,1,-1, -1, 1, 1] #vrednosti naelektrisanja
    lokacije = [[2,2],[3,0.3],[0.7,0.3], [-4,0.3], [-5,-5], [3,3]] #pocetne lokacije

    #crtamo kruzice tako da je velicina proporcionalna naelektrisanju
    #ako je q negativno onda crtamo plavi kruzic

    for i in range(len(q)):
        r=np.sqrt(np.abs(q[i])/40) #skaliram precnik kruga
        if q[i] < 0:
            abv=patches.Circle(lokacije[i], r, fc='b', alpha=0.5, picker=True)
            circles.append(ax.add_patch(abv))
        else:
            zxc=patches.Circle(lokacije[i], r, fc='r', alpha=0.5, picker=True)
            circles.append(ax.add_patch(zxc))

    drs = []

    i = 0

    for c in circles:
        dr = DraggablePoint(c, q[i], i) #inicijalizujem telo
        dr.connect() #dajem mogucnost "pokretanja"
        drs.append(dr) #pakujem u listu
        i += 1

    racun = polje(drs, X, Y) #incijalno polje
    ax.quiver(X,Y,racun[0],racun[1], color='r',  alpha=0.5)
    ax.quiver(X,Y,racun[0],racun[1], edgecolor='k', facecolor='None', linewidth =.5) #inicijalni crtez
    plt.show()
