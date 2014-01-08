#include "stdafx.h"

#include <random>
#include <time.h>

// See http://stackoverflow.com/questions/8449234/c-random-number-between-2-numbers-reset
int random(int min, int max)
{
	static bool init = false;

	if (!init) {
		srand(time(NULL));
		init = true;
	}

	return rand()%(max-min)+min; 
}