#include "opencv2\opencv.hpp"

#include <dlib\opencv.h>
#include <dlib\image_processing.h>
#include <windows.h>
#include <conio.h>

const double PI = 3.141592653589793238463;    //value of pi

using namespace cv;
using namespace dnn;
using namespace std;


vector<Point2f> punkty_na_obrazie;
Mat obraz_z_kamery;
Mat frame;

bool show_circles;
void klikniecie_mysza(int event, int x, int y, int, void *rysuj) {
	if (event == EVENT_LBUTTONDOWN)
	{
		show_circles = true;
	}
	if (event == EVENT_RBUTTONDOWN)
	{
		show_circles = false;
	}
}


int detekcja_twarzy()
{
	
	int spanko = 0;
	//Funkcja wykrywa twarze korzystaj¹c z detektora twarzy opartego o g³êbokie sieci neuronowe, 
	//nastêpnie, dla ka¿dej twarzy wykrywa jej 68 charakterystycznych punktów - ta czêœæ realizowana jest z pomoc¹ biblioteki dlib, 
	//gdy¿ rozwi¹zanie w OpenCV jest bardzo niestabilne

	// stworzenie i za³adowanie detektora twarzy
	Net net = readNetFromTensorflow("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt");

	//aby biblioteki OpenCV i dlib siê nam nie pomiesza³y, za ka¿dym razem gdy bêdziemy u¿ywaæ dlib, bêdziemy jawnie odwo³ywaæ siê do przestrzeni nazw tej biblioteki

	//stworzenie i za³adowanie detektora punktów charakterystycznych twarzy
	dlib::shape_predictor d_sp;
	dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> d_sp;
	VideoCapture kamera(0, CAP_DSHOW);
	if (!kamera.isOpened()) return 0;
	//ustalenie rozdzielczosci
	kamera.set(CAP_PROP_FRAME_WIDTH, 1280);
	kamera.set(CAP_PROP_FRAME_HEIGHT, 720);

	Mat macierzKamery = (Mat_<double>(3, 3) << 1171, 0, 663.9, 0, 1171, 328, 0, 0, 1);
	Mat wspolczynnikiZnieksztalcen = (Mat_<double>(1, 4) << 0, 0, 0, 0);
	while (waitKey(1) != 27) //jesli nie nacisnieto ESC obraz jest pobierany i wyswietlany w petli
	{
		kamera >> obraz_z_kamery;
		if (obraz_z_kamery.data == NULL) break; //zabezpieczenie - jesli nie ma nowej klatki, przerywa dzialanie
		putText(obraz_z_kamery, String("Nacisnij ESC by wylaczyc"), Point(0, 25), 0, 1, CV_RGB(255, 0, 0), 2);


		//dlib ma w³asny format obrazu, wiêc ten pobrany przez OpenCV musimy przekonwertowaæ
		//zmienne dla biblioteki dlib bêdziemy oznaczaæ przedrostkiem d_, ¿eby siê nam nie pomyli³y
		dlib::cv_image<dlib::bgr_pixel> d_obraz(obraz_z_kamery);

		//przetworzenie obrazu z kamery na format dostosowany do g³êbokiej sieci neuronowej wykrywaj¹cej twarze
		Mat blob = cv::dnn::blobFromImage(obraz_z_kamery, 1.0, Size(300, 300), Scalar(104.0, 177.0, 123.0), true, false );

		// uruchomienie g³êbokiej sieci neuronowej wykrywaj¹cej twarze
		net.setInput(blob, "data");
		Mat detekcje = net.forward("detection_out");
		// przeformatowanie wyniku detekcji: macierz "twarze" ma tyle wierszy ile wykryto twarzy,
		// w ka¿dym wierszu znajduj¹ siê kolejno:
		// wiod¹ce 0,
		// nr klasy wykrytego obiektu (ta sieæ wykrywa jeden obiekt - twarz, wiêc tam zawsze jest 1),
		// pewnoœæ wykrycia w skali 0-1,
		// znormalizowane wzglêdem wymiarów obrazu wspó³rzêdne 2 wierzcho³ków prostok¹ta zawieraj¹cego obiekt (x1, y1, x2, y2)
		Mat twarze(detekcje.size[2], detekcje.size[3], CV_32F, detekcje.ptr<float>());

		int ile = 0;
		
		if (twarze.rows > 0) //jeœli wykryto przynajmniej 1 twarz
		{
			for (int i = 0; i < twarze.rows; i++) {
				if (twarze.at<float>(i, 2) > 0.5) //pewnoœæ wykrycia (w skali 0-1), pomijamy ma³o prawdopodobne twarze
				{

					// odczyt punktów tworz¹cych prostok¹t wokó³ twarzy
					int x1 = (twarze.at<float>(i, 3) * obraz_z_kamery.cols);
					int y1 = (twarze.at<float>(i, 4) * obraz_z_kamery.rows);
					int x2 = (twarze.at<float>(i, 5) * obraz_z_kamery.cols);
					int y2 = (twarze.at<float>(i, 6) * obraz_z_kamery.rows);
					// zbudowanie prostok¹ta zawieraj¹cego twarz
					Rect twarz = Rect(Point(x1, y1), Point(x2, y2));

					//dodaæ kod rysuj¹cy prostok¹t wokó³ twarzy
					rectangle(obraz_z_kamery, twarz, CV_RGB(0, 255, 100), 2);


					//taki sam prostokat, ale w formacie biblioteki dlib
					dlib::rectangle d_twarz = dlib::rectangle(dlib::point(x1, y1), dlib::point(x2, y2));
					//uruchomienia detektora punktów dla danej twarzy
					dlib::full_object_detection d_punkty_na_twarzy = d_sp(d_obraz, d_twarz);

					cout << "Wykryto " << d_punkty_na_twarzy.num_parts() << " punktow na twarzy nr 0" << endl;
					setMouseCallback("obraz", klikniecie_mysza, (void*)true); //ustawienie funkcji czytajacej mysz na danym oknie z obrazem (true oznacza, ze rysowane sa koleczka w miejscu klikniecia)

					

					//narysowaæ wszystkie punkty na twarzy za pomoc¹ ma³ych kó³eczek a obok nich wypisaæ ich numery
					putText(obraz_z_kamery, String("LPM-show/RPM-unshow"), Point(0, 65), 0, 1, CV_RGB(255, 0, 0), 2);
					if (GetAsyncKeyState(0x43) && show_circles == 1)
						show_circles == 0;
					else if (GetAsyncKeyState(0x43) && show_circles == 0)
						show_circles == 1;
					cout << "show_circles " << show_circles << "      " << GetAsyncKeyState(0x43) << endl;

					for (int i = 0; i < d_punkty_na_twarzy.num_parts();i++)
					{
						int x = d_punkty_na_twarzy.part(i).x();
						int y = d_punkty_na_twarzy.part(i).y();
						circle(obraz_z_kamery, Point(x, y), 1, Scalar(255, 0, 255), 3, LINE_AA);
						if(show_circles)
							putText(obraz_z_kamery, to_string(i), Point(d_punkty_na_twarzy.part(i).x(), d_punkty_na_twarzy.part(i).y()), 0, 0.5, CV_RGB(255, 0, 0), 1);
						imshow("obraz", obraz_z_kamery);
					}

					//po³¹czyæ oddzielnymi liniami punkty jak na rys: http://blog.dlib.net/2014/08/real-time-face-pose-estimation.html
					for (int i = 0; i <= 15;i++) // zarys twarzy
					{
						int x = d_punkty_na_twarzy.part(i).x();
						int y = d_punkty_na_twarzy.part(i).y();
						int x1 = d_punkty_na_twarzy.part(i + 1).x();
						int y1 = d_punkty_na_twarzy.part(i + 1).y();
						line(obraz_z_kamery, Point(x, y), Point(x1, y1), CV_RGB(0, 100, 100), 2);
					}
					for (int i = 17; i <= 20;i++) // lewa brew
					{
						int x = d_punkty_na_twarzy.part(i).x();
						int y = d_punkty_na_twarzy.part(i).y();
						int x1 = d_punkty_na_twarzy.part(i + 1).x();
						int y1 = d_punkty_na_twarzy.part(i + 1).y();
						line(obraz_z_kamery, Point(x, y), Point(x1, y1), CV_RGB(0, 0, 0), 4);
					}
					for (int i = 22; i <= 25;i++) // prawa brew
					{
						int x = d_punkty_na_twarzy.part(i).x();
						int y = d_punkty_na_twarzy.part(i).y();
						int x1 = d_punkty_na_twarzy.part(i + 1).x();
						int y1 = d_punkty_na_twarzy.part(i + 1).y();
						line(obraz_z_kamery, Point(x, y), Point(x1, y1), CV_RGB(0, 0, 0), 4);
					}
					for (int i = 27; i <= 34;i++)  // nos 
					{
						int x = d_punkty_na_twarzy.part(i).x();
						int y = d_punkty_na_twarzy.part(i).y();
						int x1 = d_punkty_na_twarzy.part(i + 1).x();
						int y1 = d_punkty_na_twarzy.part(i + 1).y();
						line(obraz_z_kamery, Point(x, y), Point(x1, y1), CV_RGB(0, 255, 100), 2);
					}
					line(obraz_z_kamery, Point(d_punkty_na_twarzy.part(35).x(), d_punkty_na_twarzy.part(35).y()), Point(d_punkty_na_twarzy.part(30).x(), d_punkty_na_twarzy.part(30).y()), CV_RGB(0, 255, 100), 2);

					for (int i = 36; i <= 40;i++) // lewe gorne oko
					{
						int x = d_punkty_na_twarzy.part(i).x();
						int y = d_punkty_na_twarzy.part(i).y();
						int x1 = d_punkty_na_twarzy.part(i + 1).x();
						int y1 = d_punkty_na_twarzy.part(i + 1).y();
						line(obraz_z_kamery, Point(x, y), Point(x1, y1), CV_RGB(150, 100, 0), 2);
					}
					line(obraz_z_kamery, Point(d_punkty_na_twarzy.part(36).x(), d_punkty_na_twarzy.part(36).y()), Point(d_punkty_na_twarzy.part(41).x(), d_punkty_na_twarzy.part(41).y()), CV_RGB(150, 100, 0), 2);
					for (int i = 42; i <= 46;i++) // prawe gorne oko
					{
						int x = d_punkty_na_twarzy.part(i).x();
						int y = d_punkty_na_twarzy.part(i).y();
						int x1 = d_punkty_na_twarzy.part(i + 1).x();
						int y1 = d_punkty_na_twarzy.part(i + 1).y();
						line(obraz_z_kamery, Point(x, y), Point(x1, y1), CV_RGB(150, 100, 0), 2);
					}
					line(obraz_z_kamery, Point(d_punkty_na_twarzy.part(42).x(), d_punkty_na_twarzy.part(42).y()), Point(d_punkty_na_twarzy.part(47).x(), d_punkty_na_twarzy.part(47).y()), CV_RGB(150, 100, 0), 2);


					for (int i = 48; i <= 58;i++) // zewnetrzne usta
					{
						int x = d_punkty_na_twarzy.part(i).x();
						int y = d_punkty_na_twarzy.part(i).y();
						int x1 = d_punkty_na_twarzy.part(i + 1).x();
						int y1 = d_punkty_na_twarzy.part(i + 1).y();
						line(obraz_z_kamery, Point(x, y), Point(x1, y1), CV_RGB(255, 0, 0), 2);
					}
					line(obraz_z_kamery, Point(d_punkty_na_twarzy.part(48).x(), d_punkty_na_twarzy.part(48).y()), Point(d_punkty_na_twarzy.part(59).x(), d_punkty_na_twarzy.part(59).y()), CV_RGB(255, 0, 0), 2);

					for (int i = 60; i <= d_punkty_na_twarzy.num_parts() - 2;i++) // wewnetrzne usta
					{
						int x = d_punkty_na_twarzy.part(i).x();
						int y = d_punkty_na_twarzy.part(i).y();
						int x1 = d_punkty_na_twarzy.part(i + 1).x();
						int y1 = d_punkty_na_twarzy.part(i + 1).y();
						line(obraz_z_kamery, Point(x, y), Point(x1, y1), CV_RGB(255, 0, 0), 2);
					}
					line(obraz_z_kamery, Point(d_punkty_na_twarzy.part(60).x(), d_punkty_na_twarzy.part(60).y()), Point(d_punkty_na_twarzy.part(67).x(), d_punkty_na_twarzy.part(67).y()), CV_RGB(255, 0, 0), 2);


					//WYZNACZENIE POZYCJI I ORIENTACJI G£OWY
					//punkty na obrazie mamy, potrzebne s¹ punkty na obiekcie - w przybli¿eniu mo¿na przyj¹æ poni¿sze (w mm)
					//dla lepszej dok³adnoœci mo¿na siê zmierzyæ linijk¹
					/*Czubek nosa : (0, 0, 0)
					Broda : (0, -80, -13)
					Lewy k¹cik lewego oka : (50, 30, -50)
					Prawy k¹cik prawego oka : (-50, 30, -50)
					Lewy k¹cik ust : (30, -35, -25)
					Prawy k¹cik ust : (-30 -35, -25)*/

					// lewy srodek brwi
					// prawy srodek brwi

					//vector<Point3f>  punkty_na_obiekcie{ { 0, 0, 0 }, { 0, -84, -16 }, { 57, 46, -34 }, {-57, 46, -34 }, {38, -38, -32}, {-38, -38, -32 },{15,50,-18},{-15,50,-18} }; //uzupe³niæ
					vector<Point3f>  punkty_na_obiekcie{ { 0, 0, 0 }, {0, -80, -13}, {50, 30, -50 }, { -50, 30, -50 }, { 30, -35, -25 }, { -30, -35, -25 } };
					vector<Point2f>  punkty_na_obrazie;

					// wrzuciæ odpowiednie punkty w odpowiedniej kolejnoœci z wykrytej twarzy
					if (d_punkty_na_twarzy.num_parts() > 0) //je¿eli wykryto punkty
					{
						//wype³niæ wektor punktów na obrazie, przepisuj¹c wspó³rzêdne z odpowiednich wykrytych punktów, jak w przyk³adzie poni¿ej
						punkty_na_obrazie.push_back(Point(d_punkty_na_twarzy.part(30).x(), d_punkty_na_twarzy.part(30).y())); // czubek nosa
						punkty_na_obrazie.push_back(Point(d_punkty_na_twarzy.part(8).x(), d_punkty_na_twarzy.part(8).y())); // broda
						punkty_na_obrazie.push_back(Point(d_punkty_na_twarzy.part(45).x(), d_punkty_na_twarzy.part(45).y())); // lewy kacik lewego oka
						punkty_na_obrazie.push_back(Point(d_punkty_na_twarzy.part(36).x(), d_punkty_na_twarzy.part(36).y())); // prawy kacik prawego oka
						punkty_na_obrazie.push_back(Point(d_punkty_na_twarzy.part(54).x(), d_punkty_na_twarzy.part(54).y())); // lewy kacik ust
						punkty_na_obrazie.push_back(Point(d_punkty_na_twarzy.part(48).x(), d_punkty_na_twarzy.part(48).y())); // prawy kacik ust

						//punkty_na_obrazie.push_back(Point(d_punkty_na_twarzy.part(22).x(), d_punkty_na_twarzy.part(22).y())); // lewa brew srodek k¹cik
						//punkty_na_obrazie.push_back(Point(d_punkty_na_twarzy.part(21).x(), d_punkty_na_twarzy.part(21).y())); // prawa brew srodek k¹cik

						//narysuj te punkty wyraŸnymi kó³kami
						circle(obraz_z_kamery, Point(d_punkty_na_twarzy.part(30).x(), d_punkty_na_twarzy.part(30).y()), 2, Scalar(255, 0, 0), 3, LINE_AA);	// czubek nosa
						circle(obraz_z_kamery, Point(d_punkty_na_twarzy.part(8).x(), d_punkty_na_twarzy.part(8).y()), 2, Scalar(255, 0, 0), 3, LINE_AA);	// broda
						circle(obraz_z_kamery, Point(d_punkty_na_twarzy.part(45).x(), d_punkty_na_twarzy.part(45).y()), 2, Scalar(255, 0, 0), 3, LINE_AA);	// lewy kacik lewego oka
						circle(obraz_z_kamery, Point(d_punkty_na_twarzy.part(36).x(), d_punkty_na_twarzy.part(36).y()), 2, Scalar(255, 0, 0), 3, LINE_AA);	// prawy kacik prawego oka
						circle(obraz_z_kamery, Point(d_punkty_na_twarzy.part(54).x(), d_punkty_na_twarzy.part(54).y()), 2, Scalar(255, 0, 0), 3, LINE_AA);	 // lewy kacik ust
						circle(obraz_z_kamery, Point(d_punkty_na_twarzy.part(48).x(), d_punkty_na_twarzy.part(48).y()), 2, Scalar(255, 0, 0), 3, LINE_AA);	// prawy kacik ust

						//circle(obraz_z_kamery, Point(d_punkty_na_twarzy.part(22).x(), d_punkty_na_twarzy.part(22).y()), 2, Scalar(255, 0, 0), 3, LINE_AA);	// lewa brew srodek k¹cik
						//circle(obraz_z_kamery, Point(d_punkty_na_twarzy.part(21).x(), d_punkty_na_twarzy.part(21).y()), 2, Scalar(255, 0, 0), 3, LINE_AA);	// prawa brew srodek k¹cik

						//maj¹c wspó³rzêdne obiektowe i ekranowe, wyznacz transformacjê
						//Mat r;   // 0,0,500, a wektor rotacji -pi, 0, 0
						Mat r = (Mat_<double>(3, 1) << -PI, 0, 0);
						Mat t = (Mat_<double>(3, 1) << 0, 0, 700);


						solvePnP(punkty_na_obiekcie, punkty_na_obrazie, macierzKamery, wspolczynnikiZnieksztalcen, r, t, true, SolvePnPMethod::SOLVEPNP_ITERATIVE);
						Mat T_ko = Mat::eye(4, 4, CV_64F); //macierz jednostkowa
						Rodrigues(r, T_ko(Rect(0, 0, 3, 3))); //przeksztalcenie wektora rotacji na klasyczna macierz rotacji i skopiowanie jej do macierzy transformacji uogolnionej
						Mat(t).copyTo(T_ko(Rect(3, 0, 1, 3))); //skopiowanie wektora translacji do macierzy T
						cout << T_ko << endl;
						cout << r << endl;
						cout << t << endl;
						vector<Point3f> strzalki3d{ { 0, 0, 0 },{ 80, 0, 0 },{ 0, 80, 0 },{ 0, 0, 80 } };
						vector<Point2f> strzalki2d; //tu znajda sie wspolrzedne EKRANOWE tych 4 punktow powyzej
						projectPoints(strzalki3d, r, t, macierzKamery, wspolczynnikiZnieksztalcen, strzalki2d);


						//i narysuj oœ Z

						//teoretycznie powinna zaczynaæ siê na czubku nosa, ale ze wzglêdu na du¿e niedok³adnoœci
						//pomiêdzy punktami 3D a rzeczywistymi wymiarami g³owy, oœ mo¿e wyraŸnie skakaæ
						//mo¿na j¹ wówczas na si³ê narysowaæ z czubka nosa - wszak mamy jego wspó³rzêdne ekranowe!
						//line(obraz_z_kamery, Point(d_punkty_na_twarzy.part(30).x(), d_punkty_na_twarzy.part(30).y()), strzalki2d[1], CV_RGB(255, 0, 0), 2);
						//line(obraz_z_kamery, Point(d_punkty_na_twarzy.part(30).x(), d_punkty_na_twarzy.part(30).y()), strzalki2d[2], CV_RGB(0, 255, 0), 2);
						//line(obraz_z_kamery, Point(d_punkty_na_twarzy.part(30).x(), d_punkty_na_twarzy.part(30).y()), strzalki2d[3], CV_RGB(0, 0, 255), 2);

						//narysuj uk³ad wspó³rzêdnych twarzy - mo¿na do tego u¿yæ gotowej funkcji
						drawFrameAxes(obraz_z_kamery,macierzKamery,wspolczynnikiZnieksztalcen,r,t,80);

						//zdefiniuj 8 punktów 3D pude³ka otaczaj¹cego g³owê i narysuj pude³ko analogicznie jak w poprzednim projekcie rysowaliœmy osie na szachownicach
						vector<Point3f> pudlo3d{ { 50, -84, 0 }, {50, -84, -150 }, {-50, -84, -150 }, {-50, -84, 0}, { 50, 100, 0 }, {50, 100, -150 }, {-50, 100, -150 }, {-50, 100, 0} }; // z przodu to 0,3,4,7
						vector<Point2f> pudlo2d;
						projectPoints(pudlo3d, r, t, macierzKamery, wspolczynnikiZnieksztalcen, pudlo2d);

						line(obraz_z_kamery, pudlo2d[0], pudlo2d[1], CV_RGB(150, 150, 150), 2);
						line(obraz_z_kamery, pudlo2d[1], pudlo2d[2], CV_RGB(100, 100, 100), 2);
						line(obraz_z_kamery, pudlo2d[2], pudlo2d[3], CV_RGB(150, 150, 150), 2);
						line(obraz_z_kamery, pudlo2d[3], pudlo2d[0], CV_RGB(200, 200, 200), 2);
						line(obraz_z_kamery, pudlo2d[4], pudlo2d[5], CV_RGB(150, 150, 150), 2);
						line(obraz_z_kamery, pudlo2d[5], pudlo2d[6], CV_RGB(100, 100, 100), 2);
						line(obraz_z_kamery, pudlo2d[6], pudlo2d[7], CV_RGB(150, 150, 150), 2);
						line(obraz_z_kamery, pudlo2d[7], pudlo2d[4], CV_RGB(200, 200, 200), 2);
						line(obraz_z_kamery, pudlo2d[0], pudlo2d[4], CV_RGB(200, 200, 200), 2);
						line(obraz_z_kamery, pudlo2d[1], pudlo2d[5], CV_RGB(100, 100, 100), 2);
						line(obraz_z_kamery, pudlo2d[2], pudlo2d[6], CV_RGB(100, 100, 100), 2);
						line(obraz_z_kamery, pudlo2d[3], pudlo2d[7], CV_RGB(200, 200, 200), 2);

						//aby poprawiæ jakoœæ dopasowania modelu i pozbyæ siê irytuj¹cych drgañ i obrotów pud³a, podaj komendzie solvePnP przybli¿one parametry poszukiwanej transformacji
						//wektor translacji bêdzie w przybli¿eniu: 0,0,500, a wektor rotacji -pi, 0, 0
						//wystarczy zainicjowaæ te wektory w ten sposób oraz dodaæ parametr "true" na koñcu funkcji solvePnP
						//dziêki temu funkcja znajdzie optimum w pobli¿u tego punktu, a nie w innym miejscu, co objawia siê z³ym dopasowaniem pud³a

						//wyznacz i wypisz na ekranie k¹ty RPY
						double beta = atan2(-T_ko.at<double>(2, 0), sqrt(pow(T_ko.at<double>(0, 0), 2) + pow(T_ko.at<double>(1, 0), 2)));
						double alfa = atan2(T_ko.at<double>(1, 0) / cos(beta), T_ko.at<double>(0, 0) / cos(beta));
						double gamma = atan2(T_ko.at<double>(2, 1) / cos(beta), T_ko.at<double>(2, 2) / cos(beta));
						//na podstawie odpowiedniego k¹ta stwierdŸ czy g³owa nie opad³a do przodu lub do ty³u - znaczy kierowca œpi
						cout << "beta = " << beta * 180 / 3.14 << endl;
						cout << "alfa = " << alfa * 180 / 3.14 << endl; // jesli beta sie zmieni == kierowca spi
						cout << "gamma = " << gamma * 180 / 3.14 << endl;

						if (alfa * 180 / 3.14 > 15) {
							putText(obraz_z_kamery, String("glowa w lewo"), Point(0, 115), 0, 1, CV_RGB(255, 0, 0), 2);
						}
						if (alfa * 180 / 3.14 < -15) {
							putText(obraz_z_kamery, String("glowa w prawo"), Point(0, 115), 0, 1, CV_RGB(255, 0, 0), 2);
						}
						if (gamma * 180 / 3.14 < 175 && gamma * 180 / 3.14 >100) {
							putText(obraz_z_kamery, String("glowa w gore"), Point(0, 145), 0, 1, CV_RGB(255, 0, 0), 2);
						}
						if (gamma * 180 / 3.14 < 0 && gamma * 180 / 3.14 > -159) {
							putText(obraz_z_kamery, String("glowa w dol"), Point(0, 145), 0, 1, CV_RGB(255, 0, 0), 2);
						}

						//niektórzy jednak œpi¹ z g³ow¹ prosto - sprawdŸ czy aby nie zamknêli oczu

						//aby nie by³o fa³szywych alarmów, stan zaœniêcia musi trwaæ bez przerwy przez 30 klatek
						float prawa_szerokosc_oka = norm(Point(d_punkty_na_twarzy.part(36).x(), d_punkty_na_twarzy.part(36).y()) - Point(d_punkty_na_twarzy.part(39).x(), d_punkty_na_twarzy.part(39).y()));
						float prawa_wysokosc_oka = norm(Point(d_punkty_na_twarzy.part(37).x(), d_punkty_na_twarzy.part(37).y()) - Point(d_punkty_na_twarzy.part(41).x(), d_punkty_na_twarzy.part(41).y()));;
						float prawy_stosunek = prawa_szerokosc_oka / prawa_wysokosc_oka;
						float lewa_szerokosc_oka = norm(Point(d_punkty_na_twarzy.part(42).x(), d_punkty_na_twarzy.part(42).y()) - Point(d_punkty_na_twarzy.part(45).x(), d_punkty_na_twarzy.part(45).y()));;
						float lewa_wysokosc_oka = norm(Point(d_punkty_na_twarzy.part(47).x(), d_punkty_na_twarzy.part(47).y()) - Point(d_punkty_na_twarzy.part(43).x(), d_punkty_na_twarzy.part(43).y()));;
						float lewy_stosunek = lewa_szerokosc_oka / lewa_wysokosc_oka;
						cout << prawy_stosunek << endl;
						cout << lewy_stosunek << endl;

						if (prawy_stosunek > 3.0 &&lewy_stosunek > 3.0)
							spanko++;
						else spanko = 0;
						if (spanko > 15)
							putText(obraz_z_kamery, String("Obudz sie-otworz oczy"), Point(0, 105), 0, 1, CV_RGB(255, 0, 0), 2);
						cout << "spanko = " << spanko << endl;
					}
				}
			}
			imshow("obraz", obraz_z_kamery);
		}
	}


	return 0;
}

int main()
{
	detekcja_twarzy();
	

	return 0;
}


