/* basé sur on smallpt, a Path Tracer by Kevin Beason, 2008
 *  	http://www.kevinbeason.com/smallpt/ 
 *
 * Converti en C et modifié par Charles Bouillaguet, 2019
 *
 * Pour des détails sur le processus de rendu, lire :
 * 	https://docs.google.com/open?id=0B8g97JkuSSBwUENiWTJXeGtTOHFmSm51UC01YWtCZw
 */

#define _XOPEN_SOURCE
#include <math.h>   
#include <stdlib.h> 
#include <stdio.h>
#include <stdbool.h>
#include <sys/time.h>
#include <sys/stat.h>  /* pour mkdir    */ 
#include <unistd.h>    /* pour getuid   */
#include <sys/types.h> /* pour getpwuid */
#include <pwd.h>       /* pour getpwuid */
#include <mpi.h>




//Structure pour le jeton

  	typedef enum msgJeton msgJeton;
  	enum msgJeton 
  	{
  		VIDE, TRAVAIL, FINI   // Respectivement 0 1 et 42
  	};

  	typedef struct Jeton Jeton;
  	struct Jeton
  	{
  		msgJeton msg;
  		int travailleurVolontaire;

  	};


enum Refl_t {DIFF, SPEC, REFR};   /* types de matériaux (DIFFuse, SPECular, REFRactive) */

struct Sphere { 
	double radius; 
	double position[3];
	double emission[3];     /* couleur émise (=source de lumière) */
	double color[3];        /* couleur de l'objet RGB (diffusion, refraction, ...) */
	enum Refl_t refl;       /* type de reflection */
	double max_reflexivity;
};



static const int KILL_DEPTH = 7;
static const int SPLIT_DEPTH = 4;

/* la scène est composée uniquement de spheres */
struct Sphere spheres[] = { 
// radius position,                         emission,     color,              material 
   {1e5,  { 1e5+1,  40.8,       81.6},      {},           {.75,  .25,  .25},  DIFF, -1}, // Left 
   {1e5,  {-1e5+99, 40.8,       81.6},      {},           {.25,  .25,  .75},  DIFF, -1}, // Right 
   {1e5,  {50,      40.8,       1e5},       {},           {.75,  .75,  .75},  DIFF, -1}, // Back 
   {1e5,  {50,      40.8,      -1e5 + 170}, {},           {},                 DIFF, -1}, // Front 
   {1e5,  {50,      1e5,        81.6},      {},           {0.75, .75,  .75},  DIFF, -1}, // Bottom 
   {1e5,  {50,     -1e5 + 81.6, 81.6},      {},           {0.75, .75,  .75},  DIFF, -1}, // Top 


   {16.5, {40,      16.5,       47},        {},           {.999, .999, .999}, SPEC, -1}, // Mirror 
   {16.5, {73,      46.5,       88},        {},           {.999, .999, .999}, REFR, -1}, // Glass 
   {10,   {15,      45,         112},       {},           {.999, .999, .999}, DIFF, -1}, // white ball
   {15,   {16,      16,         130},       {},           {.999, .999, 0},    REFR, -1}, // big yellow glass
   {7.5,  {40,      8,          120},        {},           {.999, .999, 0   }, REFR, -1}, // small yellow glass middle
   {8.5,  {60,      9,          110},        {},           {.999, .999, 0   }, REFR, -1}, // small yellow glass right

   {10,   {80,      12,         92},        {},           {0, .999, 0},       DIFF, -1}, // green ball




   {600,  {50,      681.33,     81.6},      {12, 12, 12}, {},                 DIFF, -1},  // Light 
   {5,    {50,      75,         81.6},      {},           {0, .682, .999}, DIFF, -1}, // occlusion, mirror
}; 


//////////////////////////////////////////// AVANT LE MAIN ///////////////////////////////////////////////////////////////////////


/********** micro BLAS LEVEL-1 + quelques fonctions non-standard **************/
static inline void copy(const double *x, double *y)
{
	for (int i = 0; i < 3; i++)
		y[i] = x[i];
} 

static inline void zero(double *x)
{
	for (int i = 0; i < 3; i++)
		x[i] = 0;
} 

static inline void axpy(double alpha, const double *x, double *y)
{
	for (int i = 0; i < 3; i++)
		y[i] += alpha * x[i];
} 

static inline void scal(double alpha, double *x)
{
	for (int i = 0; i < 3; i++)
		x[i] *= alpha;
} 

static inline double dot(const double *a, const double *b)
{ 
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
} 

static inline double nrm2(const double *a)
{
	return sqrt(dot(a, a));
}

/********* fonction non-standard *************/
static inline void mul(const double *x, const double *y, double *z)
{
	for (int i = 0; i < 3; i++)
		z[i] = x[i] * y[i];
} 

static inline void normalize(double *x)
{
	scal(1 / nrm2(x), x);
}

/* produit vectoriel */
static inline void cross(const double *a, const double *b, double *c)
{
	c[0] = a[1] * b[2] - a[2] * b[1];
	c[1] = a[2] * b[0] - a[0] * b[2];
	c[2] = a[0] * b[1] - a[1] * b[0];
}

/****** tronque *************/
static inline void clamp(double *x)
{
	for (int i = 0; i < 3; i++) {
		if (x[i] < 0)
			x[i] = 0;
		if (x[i] > 1)
			x[i] = 1;
	}
} 

/******************************* calcul des intersections rayon / sphere *************************************/
   
// returns distance, 0 if nohit 
double sphere_intersect(const struct Sphere *s, const double *ray_origin, const double *ray_direction)
{ 
	double op[3];
	// Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0 
	copy(s->position, op);
	axpy(-1, ray_origin, op);
	double eps = 1e-4;
	double b = dot(op, ray_direction);
	double discriminant = b * b - dot(op, op) + s->radius * s->radius; 
	if (discriminant < 0)
		return 0;   /* pas d'intersection */
	else 
		discriminant = sqrt(discriminant);
	/* détermine la plus petite solution positive (i.e. point d'intersection le plus proche, mais devant nous) */
	double t = b - discriminant;
	if (t > eps) {
		return t;
	} else {
		t = b + discriminant;
		if (t > eps)
			return t;
		else
			return 0;  /* cas bizarre, racine double, etc. */
	}
}

/* détermine si le rayon intersecte l'une des spere; si oui renvoie true et fixe t, id */
bool intersect(const double *ray_origin, const double *ray_direction, double *t, int *id)
{ 
	int n = sizeof(spheres) / sizeof(struct Sphere);
	double inf = 1e20; 
	*t = inf;
	for (int i = 0; i < n; i++) {
		double d = sphere_intersect(&spheres[i], ray_origin, ray_direction);
		if ((d > 0) && (d < *t)) {
			*t = d;
			*id = i;
		} 
	}
	return *t < inf;
} 

/* calcule (dans out) la lumiance reçue par la camera sur le rayon donné */
void radiance(const double *ray_origin, const double *ray_direction, int depth, unsigned short *PRNG_state, double *out)
{ 
	int id = 0;                             // id de la sphère intersectée par le rayon
	double t;                               // distance à l'intersection
	if (!intersect(ray_origin, ray_direction, &t, &id)) {
		zero(out);    // if miss, return black 
		return; 
	}
	const struct Sphere *obj = &spheres[id];
	
	/* point d'intersection du rayon et de la sphère */
	double x[3];
	copy(ray_origin, x);
	axpy(t, ray_direction, x);
	
	/* vecteur normal à la sphere, au point d'intersection */
	double n[3];  
	copy(x, n);
	axpy(-1, obj->position, n);
	normalize(n);
	
	/* vecteur normal, orienté dans le sens opposé au rayon 
	   (vers l'extérieur si le rayon entre, vers l'intérieur s'il sort) */
	double nl[3];
	copy(n, nl);
	if (dot(n, ray_direction) > 0)
		scal(-1, nl);
	
	/* couleur de la sphere */
	double f[3];
	copy(obj->color, f);
	double p = obj->max_reflexivity;

	/* processus aléatoire : au-delà d'une certaine profondeur,
	   décide aléatoirement d'arrêter la récusion. Plus l'objet est
	   clair, plus le processus a de chance de continuer. */
	depth++;
	if (depth > KILL_DEPTH) {
		if (erand48(PRNG_state) < p) {
			scal(1 / p, f); 
		} else {
			copy(obj->emission, out);
			return;
		}
	}

	/* Cas de la réflection DIFFuse (= non-brillante). 
	   On récupère la luminance en provenance de l'ensemble de l'univers. 
	   Pour cela : (processus de monte-carlo) on choisit une direction
	   aléatoire dans un certain cone, et on récupère la luminance en 
	   provenance de cette direction. */
	if (obj->refl == DIFF) {
		double r1 = 2 * M_PI * erand48(PRNG_state);  /* angle aléatoire */
		double r2 = erand48(PRNG_state);             /* distance au centre aléatoire */
		double r2s = sqrt(r2); 
		
		double w[3];   /* vecteur normal */
		copy(nl, w);
		
		double u[3];   /* u est orthogonal à w */
		double uw[3] = {0, 0, 0};
		if (fabs(w[0]) > .1)
			uw[1] = 1;
		else
			uw[0] = 1;
		cross(uw, w, u);
		normalize(u);
		
		double v[3];   /* v est orthogonal à u et w */
		cross(w, u, v);
		
		double d[3];   /* d est le vecteur incident aléatoire, selon la bonne distribution */
		zero(d);
		axpy(cos(r1) * r2s, u, d);
		axpy(sin(r1) * r2s, v, d);
		axpy(sqrt(1 - r2), w, d);
		normalize(d);
		
		/* calcule récursivement la luminance du rayon incident */
		double rec[3];
		radiance(x, d, depth, PRNG_state, rec);
		
		/* pondère par la couleur de la sphère, prend en compte l'emissivité */
		mul(f, rec, out);
		axpy(1, obj->emission, out);
		return;
	}

	/* dans les deux autres cas (réflection parfaite / refraction), on considère le rayon
	   réfléchi par la spère */

	double reflected_dir[3];
	copy(ray_direction, reflected_dir);
	axpy(-2 * dot(n, ray_direction), n, reflected_dir);

	/* cas de la reflection SPEculaire parfaire (==mirroir) */
	if (obj->refl == SPEC) { 
		double rec[3];
		/* calcule récursivement la luminance du rayon réflechi */
		radiance(x, reflected_dir, depth, PRNG_state, rec);
		/* pondère par la couleur de la sphère, prend en compte l'emissivité */
		mul(f, rec, out);
		axpy(1, obj->emission, out);
		return;
	}

	/* cas des surfaces diélectriques (==verre). Combinaison de réflection et de réfraction. */
	bool into = dot(n, nl) > 0;      /* vient-il de l'extérieur ? */
	double nc = 1;                   /* indice de réfraction de l'air */
	double nt = 1.5;                 /* indice de réfraction du verre */
	double nnt = into ? (nc / nt) : (nt / nc);
	double ddn = dot(ray_direction, nl);
	
	/* si le rayon essaye de sortir de l'objet en verre avec un angle incident trop faible,
	   il rebondit entièrement */
	double cos2t = 1 - nnt * nnt * (1 - ddn * ddn);
	if (cos2t < 0) {
		double rec[3];
		/* calcule seulement le rayon réfléchi */
		radiance(x, reflected_dir, depth, PRNG_state, rec);
		mul(f, rec, out);
		axpy(1, obj->emission, out);
		return;
	}
	
	/* calcule la direction du rayon réfracté */
	double tdir[3];
	zero(tdir);
	axpy(nnt, ray_direction, tdir);
	axpy(-(into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)), n, tdir);

	/* calcul de la réflectance (==fraction de la lumière réfléchie) */
	double a = nt - nc;
	double b = nt + nc;
	double R0 = a * a / (b * b);
	double c = 1 - (into ? -ddn : dot(tdir, n));
	double Re = R0 + (1 - R0) * c * c * c * c * c;   /* réflectance */
	double Tr = 1 - Re;                              /* transmittance */
	
	/* au-dela d'une certaine profondeur, on choisit aléatoirement si
	   on calcule le rayon réfléchi ou bien le rayon réfracté. En dessous du
	   seuil, on calcule les deux. */
	double rec[3];
	if (depth > SPLIT_DEPTH) {
		double P = .25 + .5 * Re;             /* probabilité de réflection */
		if (erand48(PRNG_state) < P) {
			radiance(x, reflected_dir, depth, PRNG_state, rec);
			double RP = Re / P;
			scal(RP, rec);
		} else {
			radiance(x, tdir, depth, PRNG_state, rec);
			double TP = Tr / (1 - P); 
			scal(TP, rec);
		}
	} else {
		double rec_re[3], rec_tr[3];
		radiance(x, reflected_dir, depth, PRNG_state, rec_re);
		radiance(x, tdir, depth, PRNG_state, rec_tr);
		zero(rec);
		axpy(Re, rec_re, rec);
		axpy(Tr, rec_tr, rec);
	}
	/* pondère, prend en compte la luminance */
	mul(f, rec, out);
	axpy(1, obj->emission, out);
	return;
}

double wtime()
{
	struct timeval ts;
	gettimeofday(&ts, NULL);
	return (double)ts.tv_sec + ts.tv_usec / 1E6;
}

int toInt(double x)
{
	return pow(x, 1 / 2.2) * 255 + .5;   /* gamma correction = 2.2 */
} 

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////:


int main(int argc, char **argv)
{ 
	/* Petit cas test (small, quick and dirty): */
	//int w = 320; 
	//int h = 200; 
	//int samples = 200;

	int w = 160;
	int h = 100;
	int samples = 100;


	/* Gros cas test (big, slow and pretty): */
	/* int w = 3840; */
	/* int h = 2160; */
	/* int samples = 5000;  */


////////////////////////////////// INITIALISATION /////////////////////////////////////////



	MPI_Init(&argc,&argv);

	int rank,size;

  	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  	MPI_Comm_size(MPI_COMM_WORLD,&size);

  	MPI_Status status;

  	int ligneSuivante=size;

  	int maLigne;

  	int ligneMilieu;

  	int ligneDebut,ligneFin;

  	ligneDebut=rank*h/size;
  	ligneFin=(rank+1)*h/size;

  	int flag=-1;

  	int travailleurVolontaire;

  	int travailAFaire[2]; //={LigneDebut,LigneFin}

  	int volontariat=0; //Booléen =1 si le processus devient au moins une fois volontaire, ie qu'il a finit son travail initial

  	int envoi=0; //Booléen qui évite d'envoyer 2 messages inopinés dans le même tour de boucle

  	int employeur;

  	int travailEnvoye=0; //Booléen pour éviter que le travail ne soit envoyé 2 fois

  	int taille;

  	int limite;


  	//Création du jeton


/* Vu que c'est relou de faire communiquer des struct avec MPI on va essayer autre chose

  	Jeton jeton;

  	jeton.msg=VIDE;

  */

  	int jeton=-1;

  	// -1 : signifie qu'il n'y a rien dans le jeton
  	// -2 : signifie que le travail est terminé
  	// -3 : attend et n'envoie pas de message
  	// -4 : signifie qu'on cherche à rentrer en communication avec le processus pour lui donner du travail
  	// -5 : signifie que le processus va reçevoir le travail qu'il a demandé




  	printf("\n Rank : %d, ligne Debut=%d, ligneFin=%d \n",rank,ligneDebut,ligneFin);




//////////////////////////////////////////////////////////////////////////////////////




	if (argc == 2) 
		samples = atoi(argv[1]) / 4;

	static const double CST = 0.5135;  /* ceci défini l'angle de vue */
	double camera_position[3] = {50, 52, 295.6};
	double camera_direction[3] = {0, -0.042612, -1};
	normalize(camera_direction);

	/* incréments pour passer d'un pixel à l'autre */
	double cx[3] = {w * CST / h, 0, 0};    
	double cy[3];
	cross(cx, camera_direction, cy);  /* cy est orthogonal à cx ET à la direction dans laquelle regarde la caméra */
	normalize(cy);
	scal(CST, cy);

	/* précalcule la norme infinie des couleurs */
	int n = sizeof(spheres) / sizeof(struct Sphere);
	for (int i = 0; i < n; i++) {
		double *f = spheres[i].color;
		if ((f[0] > f[1]) && (f[0] > f[2]))
			spheres[i].max_reflexivity = f[0]; 
		else {
			if (f[1] > f[2])
				spheres[i].max_reflexivity = f[1];
			else
				spheres[i].max_reflexivity = f[2]; 
		}
	}


////////////////////////////////////////////////////////////// PARALLELISATION ////////////////////////////////////////////////////////////////////////////////


	/* boucle principale */
	//double *image = malloc(3 * w * h/size * sizeof(double));
	double *image = malloc(3 * w * h * sizeof(*image));
	if (image == NULL) {
		perror("Impossible d'allouer l'image");
		exit(1);
	}

	double *block = malloc(3 * w * h * sizeof(*block));
	if (block == NULL) {
		perror("Impossible d'allouer l'image");
		exit(1);
	}

	double *imageFinal;

	if (rank==0){

		imageFinal = (double *) malloc(3 * w * h * sizeof(double));
		if (imageFinal == NULL) {
			perror("Impossible d'allouer l'image");
			exit(1);
		}
	}

	
			

	int compteur=0;



	maLigne=ligneDebut;


	//Tant que tout n'a pas été fini
	while(jeton!=-2){

		envoi=0;




///////////////////////////////////// GESTION DU JETON ///////////////////////////////////////////////////


		MPI_Iprobe(MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&flag,&status);


		//Si le jeton est là
		if (flag==1){ 

			MPI_Recv(&jeton,1,MPI_INT,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

			printf("Rank %d a reçu le jeton=%d\n",rank,jeton);



			//Si on a reçu une demande de travail d'un autre rank que le notre
			if (jeton>=0 && jeton!=rank && envoi==0){

				//Mais qu'on est encore entrain de travail
				if (maLigne<ligneFin){

					travailleurVolontaire=jeton;

					//On supprime la demande de travail
					jeton=-1;

					if (rank==size-1){
						MPI_Send(&jeton,1,MPI_INT,0,0,MPI_COMM_WORLD);
					}

					else{
						MPI_Send(&jeton,1,MPI_INT,rank+1,0,MPI_COMM_WORLD);
					}

					//On change sa ligne de Fin
					travailAFaire[0]=(ligneFin-maLigne)/2+maLigne;
					travailAFaire[1]=ligneFin;
					


					printf("Rank %d envoit du travail à faire à rank %d\n",rank,travailleurVolontaire);

					//On dit au travailleur Volontaire qu'on va lui envoyer du travail
					jeton=-4;
					MPI_Send(&jeton,1,MPI_INT,travailleurVolontaire,0,MPI_COMM_WORLD);
					//On envoie le travail à faire
					MPI_Send(travailAFaire,2,MPI_INT,travailleurVolontaire,0,MPI_COMM_WORLD);
					jeton=-1;

					envoi=1;

					//On règle sa ligne de Fin
					ligneFin=travailAFaire[0];

				}


				//Si on a fini de travailler
				else{

					// On fait passer la demande de travail
					if (rank==size-1){
						MPI_Send(&jeton,1,MPI_INT,0,0,MPI_COMM_WORLD);
					}

					else{
						MPI_Send(&jeton,1,MPI_INT,rank+1,0,MPI_COMM_WORLD);
					}

					envoi=1;



				}


			}



			//Si on veut donner du travail au processus
			if(jeton==-4 && envoi==0){

				printf("Rank %d reçoit du travail à faire de la part de %d\n",rank,status.MPI_SOURCE);

				employeur=status.MPI_SOURCE;

				MPI_Recv(travailAFaire,2,MPI_INT,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

				ligneDebut=travailAFaire[0];
				ligneFin=travailAFaire[1];
				maLigne=ligneDebut;
				travailEnvoye=0;
				
				//On réinitialise le jeton pour éviter que le processus envoie 2 fois du travail
				jeton=-1;

			}

			/*

			//On reçoit le travail qui a été effectué
			if (jeton==-5 && envoi==0){

				MPI_Recv(&taille,1,MPI_INT,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

				MPI_Recv(&limite,1,MPI_INT,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

				//MPI_Recv(image+3*w*ligneDebut,3*w*taille,MPI_DOUBLE,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				MPI_Recv(image+3*w*limite,3*w*taille,MPI_DOUBLE,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				//MPI_Recv(image,3*w*taille,MPI_DOUBLE,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

				printf("Rank %d a reçu le travail qu'il a demandé à rank %d\n",rank,status.MPI_SOURCE);

				jeton=-1;

			}

			*/




			if (jeton==-1 && envoi==0){

				//Si on n'a pas encore fini son travail
				if (maLigne<ligneFin){


					//On renvoie -1 à tout le monde
					if (rank==size-1){
						MPI_Send(&jeton,1,MPI_INT,0,0,MPI_COMM_WORLD);
					}

					else{
						MPI_Send(&jeton,1,MPI_INT,rank+1,0,MPI_COMM_WORLD);
					}

					envoi=1;
				}

				//Si on a fini son travail
				else{

					//On envoie qu'on est dispo pour travailler
					jeton=rank;

					if (rank==size-1){
						MPI_Send(&jeton,1,MPI_INT,0,0,MPI_COMM_WORLD);
					}

					else{
						MPI_Send(&jeton,1,MPI_INT,rank+1,0,MPI_COMM_WORLD);
					}

					envoi=1;

					//On prend contact avec son employeur
				}
			}


			//Si on reçoit notre propre demande de travail
			if (jeton==rank && envoi==0){

				printf("Rank %d envoie le jeton de fin car il a reçu jeton=%d\n",rank,jeton);

				jeton=-2;

				if (rank==size-1){
					MPI_Send(&jeton,1,MPI_INT,0,0,MPI_COMM_WORLD);
				}

				else{
					MPI_Send(&jeton,1,MPI_INT,rank+1,0,MPI_COMM_WORLD);
				}

				jeton=-3; // Cela gèle le jeton, il va juste effectué des tours de boucle, jusqu'à attendre de reçevoir un jeton -2
				envoi=1;

			}


			// Si on reçoit qu'il n'y a plus de travail à faire
			if (jeton==-2 && envoi==0){

				//Si c'est le dernier processus qui reçoit le jeton, il le renvoie à rank 0
				if (rank==size-1){
					MPI_Send(&jeton,1,MPI_INT,0,0,MPI_COMM_WORLD);
				}

				else{
					MPI_Send(&jeton,1,MPI_INT,rank+1,0,MPI_COMM_WORLD);
				}

				envoi=1;
			}


		flag=-1;

		}




		

///////////////////////////////////////////////// CALCUL D'UNE LIGNE  ///////////////////////////////////////////////////////////////
 
		if (jeton!=-3 || maLigne!=ligneFin){

	 		unsigned short PRNG_state[3] = {0, 0, maLigne*maLigne*maLigne};


			for (unsigned short j = 0; j < w; j++) {
				/* calcule la luminance d'un pixel, avec sur-échantillonnage 2x2 */
				double pixel_radiance[3] = {0, 0, 0};
				for (int sub_i = 0; sub_i < 2; sub_i++) {
					for (int sub_j = 0; sub_j < 2; sub_j++) {
						double subpixel_radiance[3] = {0, 0, 0};
						/* simulation de monte-carlo : on effectue plein de lancers de rayons et on moyenne */
						for (int s = 0; s < samples; s++) { 
							/* tire un rayon aléatoire dans une zone de la caméra qui correspond à peu près au pixel à calculer */
							double r1 = 2 * erand48(PRNG_state);
							double dx = (r1 < 1) ? sqrt(r1) - 1 : 1 - sqrt(2 - r1); 
							double r2 = 2 * erand48(PRNG_state);
							double dy = (r2 < 1) ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
							double ray_direction[3];
							copy(camera_direction, ray_direction);
							axpy(((sub_i + .5 + dy) / 2 + maLigne) / h - .5, cy, ray_direction);
							axpy(((sub_j + .5 + dx) / 2 + j) / w - .5, cx, ray_direction);
							normalize(ray_direction);

							double ray_origin[3];
							copy(camera_position, ray_origin);
							axpy(140, ray_direction, ray_origin);

							/* estime la lumiance qui arrive sur la caméra par ce rayon */
							double sample_radiance[3];
							radiance(ray_origin, ray_direction, 0, PRNG_state, sample_radiance);
							/* fait la moyenne sur tous les rayons */
							axpy(1. / samples, sample_radiance, subpixel_radiance);
						}
						clamp(subpixel_radiance);
						/* fait la moyenne sur les 4 sous-pixels */
						axpy(0.25, subpixel_radiance, pixel_radiance);
					}
				}

			       // printf("\n pixel_radiance = {%d,%d,%d}\n",pixel_radiance[0],pixel_radiance[1],pixel_radiance[2]); 
				//copy(pixel_radiance, image + 3 * ((h - 1 - maLigne) * w + j)); // <-- retournement vertical
				copy(pixel_radiance, image + 3 * (((ligneFin-ligneDebut) - 1 - maLigne) * w + j)); // <-- retournement vertical
			 	//copy(pixel_radiance, image + 3 * (((h/size) - 1 - (ligneFin-maLigne)) * w + j)); // <-- retournement vertical
			       //copy(pixel_radiance,image+3*((maLigne-ligneDebut)*w+j));
	            //copy(pixel_radiance,image+3*((maLigne-ligneDebut)*w+(w-j))); //Pour inverser entre gauche et droite
	            compteur++;
			}	

			printf("Rank:%d ligne:%d \n",rank,maLigne);

		}	

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


		//printf("Rank :%d Jusqu'ici tout va bien\n",rank);

		



		//Permet à rank 0 d'envoyer le premier message
		if (maLigne==0){

			jeton=-1;

			MPI_Send(&jeton,1,MPI_INT,rank+1,0,MPI_COMM_WORLD);
		}


		//Si on n'a pas atteint la ligne de Fin
		if (maLigne<ligneFin){
		//On passe à la ligne suivante
			maLigne++;
		}


		/*
		//Le processus devient volontaire pour du travail
		if (maLigne==ligneFin){

			//Si le travail n'est pas celui initialement prévu
			if (volontariat==1 && travailEnvoye==0){

				printf("Rank %d va envoyer le travail effectué à rank %d\n",rank,employeur);

				jeton=-5;
				MPI_Send(&jeton,1,MPI_INT,employeur,0,MPI_COMM_WORLD);

				taille=travailAFaire[1]-travailAFaire[0];
				MPI_Send(&taille,1,MPI_INT,employeur,0,MPI_COMM_WORLD);

				limite=ligneFin;
				MPI_Send(&limite,1,MPI_INT,employeur,0,MPI_COMM_WORLD);

				MPI_Send(image+3*w*ligneFin,3*w*taille,MPI_DOUBLE,employeur,0,MPI_COMM_WORLD);  //Car on remplit l'img à l'envers
				//MPI_Send(image+3*w*ligneDebut,3*w*taille,MPI_DOUBLE,employeur,0,MPI_COMM_WORLD);
				//MPI_Send(image,3*w*taille,MPI_DOUBLE,employeur,0,MPI_COMM_WORLD);
				//travailEnvoye=1;

				printf("Rank %d a envoyé le travail effectué à rank %d\n",rank,employeur);
			}


			//On enregistre le travail initial dans block
			if (volontariat==0){

				int k;

				for (k=0;k<3*w*h;k++){
					block[k]=image[k];
				}
			}

			volontariat=1;
			travailEnvoye=1;
		}*/
	}
	

	printf("\nOn va passer au Gather pour rank %d\n",rank);
	//printf("w*h/size=%d\n",w*h/size);
	//printf("compteur=%d\n",3*compteur);

	

	//MPI_Gather(image,3*w*h/size,MPI_DOUBLE,imageFinal,3*w*h/size,MPI_DOUBLE,0,MPI_COMM_WORLD);
	//MPI_Gather(block,3*w*h/size,MPI_DOUBLE,imageFinal,3*w*h/size,MPI_DOUBLE,0,MPI_COMM_WORLD);

	MPI_Allreduce(MPI_IN_PLACE, image, 3*w*h, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	

	fprintf(stderr, "\n");






///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////:


	//printf("\n On va passer à l'enregistrement pour %d \n", rank);


	/* stocke l'image dans un fichier au format NetPbm */
	
	if (rank==0){

		//printf("\n Enregistrement de l'image pour %d \n",rank);

		struct passwd *pass; 
		char nom_sortie[100] = "";
		char nom_rep[30] = "";

		pass = getpwuid(getuid()); 
		//sprintf(nom_rep, "/home/sasl/eleves/main/3776597/MAIN4/HPC/Projet/%s", pass->pw_name);
		sprintf(nom_rep,"/tmp/%s",pass->pw_name);
		mkdir(nom_rep, S_IRWXU);
		sprintf(nom_sortie, "%s/image0.ppm", nom_rep);
		
		FILE *f = fopen(nom_sortie, "w");
		fprintf(f, "P3\n%d %d\n%d\n", w, h, 255); 
		for (int i = 0; i < w * h; i++) 
	  		fprintf(f,"%d %d %d ", toInt(image[3 * i]), toInt(image[3 * i + 1]), toInt(image[3 * i + 2]));
	  		//fprintf(f,"%d %d %d ", toInt(imageFinal[3 *(w*h/(rank+1)-i)]), toInt(imageFinal[3 * (w*h/(rank+1)-i)+1]), toInt(imageFinal[3 * (w*h/(rank+1)-i)+2])); 
			//fprintf(f,"%d %d %d ", toInt(image[3 *(w*h/(size)-i)]), toInt(image[3 * (w*h/(size)-i)+1]), toInt(image[3 * (w*h/(size)-i)+2]));
			//fprintf(f,"%d %d %d ", toInt(image[3 *(w*h-i)]), toInt(image[3 * (w*h-i)+1]), toInt(image[3 * (w*h-i)+2]));
			//fprintf(f,"%d %d %d ", toInt(block[3 *(w*h-i)]), toInt(block[3 * (w*h-i)+1]), toInt(block[3 * (w*h-i)+2]));
		fclose(f); 
		

		printf("\n image0.ppm enregistré \n");
		free(imageFinal);
	}

    if (rank==1){
               // printf("\n Enregistrement de l'image pour %d \n",rank);
                struct passwd *pass; 
                char nom_sortie[100] = "";
                char nom_rep[30] = "";
                pass = getpwuid(getuid()); 
                //sprintf(nom_rep, "/home/sasl/eleves/main/3776597/MAIN4/HPC/Projet/%s", pass->pw_name);
                sprintf(nom_rep,"/tmp/%s",pass->pw_name);
                mkdir(nom_rep, S_IRWXU);
                sprintf(nom_sortie, "%s/image1.ppm", nom_rep);
			//printf("\n Juste avant l'ouverture de fichier tout va bien \n");                
                FILE *g = fopen(nom_sortie, "w");
			printf("\n Là, ça va bien \n");
                fprintf(g, "P3\n%d %d\n%d\n", w, h, 255); 
		//printf("\n L'enregistrement fonctionne \n");		
                for (int i = 0; i < w * h; i++) 
                        //fprintf(g,"%d %d %d ", ligneDebut, ligneDebut, ligneDebut);
			//fprintf(f,"%d %d %d ", toInt(image[3 * i]), toInt(image[3 * i + 1]), toInt(image[3 * i + 2]));
                        //fprintf(g,"%d %d %d ", toInt(image[3 *(w*h/(size)-i)]), toInt(image[3 * (w*h/(size)-i)+1]), toInt(image[3 * (w*h/(size)-i)+2])); 
                fprintf(g,"%d %d %d ", toInt(image[3 *(w*h-i)]), toInt(image[3 * (w*h-i)+1]), toInt(image[3 * (w*h-i)+2]));
                fclose(g); 
		printf("\n image1.ppm enregistré \n");
						
               // free(imageFinal);
        }




    if (rank==2){
                //printf("\n Enregistrement de l'image pour %d \n",rank);
                struct passwd *pass; 
                char nom_sortie[100] = "";
                char nom_rep[30] = "";
                pass = getpwuid(getuid()); 
                //sprintf(nom_rep, "/home/sasl/eleves/main/3776597/MAIN4/HPC/Projet/%s", pass->pw_name);
                sprintf(nom_rep,"/tmp/%s",pass->pw_name);
                mkdir(nom_rep, S_IRWXU);
                sprintf(nom_sortie, "%s/image2.ppm", nom_rep);
		//printf("\n Juste avant l'ouverture de fichier tout va bien \n");                
                FILE *g = fopen(nom_sortie, "w");
		//printf("\n Là, ça va bien \n");
                fprintf(g, "P3\n%d %d\n%d\n", w, h, 255); 
		//printf("\n L'enregistrement fonctionne \n");		
                for (int i = 0; i < w * h; i++) 
                        //fprintf(g,"%d %d %d ", ligneDebut, ligneDebut, ligneDebut);
			fprintf(g,"%d %d %d ", toInt(image[3 * i]), toInt(image[3 * i + 1]), toInt(image[3 * i + 2]));
                        //fprintf(g,"%d %d %d ", toInt(image[3 *(w*h/(size)-i)]), toInt(image[3 * (w*h/(size)-i)+1]), toInt(image[3 * (w*h/(size)-i)+2])); 
                //fprintf(g,"%d %d %d ", toInt(image[3 *(w*h-i)]), toInt(image[3 * (w*h-i)+1]), toInt(image[3 * (w*h-i)+2]));
                fclose(g); 
		printf("\n image2.ppm enregistré \n");
						
               // free(imageFinal);
        }

    if (rank==3){
                //printf("\n Enregistrement de l'image pour %d \n",rank);
                struct passwd *pass; 
                char nom_sortie[100] = "";
                char nom_rep[30] = "";
                pass = getpwuid(getuid()); 
                //sprintf(nom_rep, "/home/sasl/eleves/main/3776597/MAIN4/HPC/Projet/%s", pass->pw_name);
                sprintf(nom_rep,"/tmp/%s",pass->pw_name);
                mkdir(nom_rep, S_IRWXU);
                sprintf(nom_sortie, "%s/image3.ppm", nom_rep);
		//printf("\n Juste avant l'ouverture de fichier tout va bien \n");                
                FILE *g = fopen(nom_sortie, "w");
		//printf("\n Là, ça va bien \n");
                fprintf(g, "P3\n%d %d\n%d\n", w, h, 255); 
		//printf("\n L'enregistrement fonctionne \n");		
                for (int i = 0; i < w * h; i++) 
                        //fprintf(g,"%d %d %d ", ligneDebut, ligneDebut, ligneDebut);
			//fprintf(f,"%d %d %d ", toInt(image[3 * i]), toInt(image[3 * i + 1]), toInt(image[3 * i + 2]));
                        //fprintf(g,"%d %d %d ", toInt(image[3 *(w*h/(rank+1)-i)]), toInt(image[3 * (w*h/(rank+1)-i)+1]), toInt(image[3 * (w*h/(rank+1)-i)+2])); 
                fprintf(g,"%d %d %d ", toInt(image[3 *(w*h-i)]), toInt(image[3 * (w*h-i)+1]), toInt(image[3 * (w*h-i)+2]));
                fclose(g); 
		printf("\n image3.ppm enregistré \n");
						
               // free(imageFinal);
        }


	free(image);



	MPI_Finalize();
//       return 0;
}