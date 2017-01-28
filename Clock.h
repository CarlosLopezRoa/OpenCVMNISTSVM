#/*---------------------------------------------------------------------------
#Program.......: ClockApp.exe
#File..........: Clock.h
#Purpose.......: definition of the Clock class.
#Author........: P. Lanza ( partially based on B. Eckel open source code see copyright notice below
#Modified......: P. Lanza
#Created.......: 11/09/2005
#Last Changed..: 26/09/2005
#Version.......: 01.01
#Copyright.....:
#License.......:
#ToDo..........:
#Note..........:
#---------------------------------------------------------------------------*/

// 26/09/2005 add the void loop function.



//: C09:Cpptime.h
// From Thinking in C++, 2nd Edition
// Available at http://www.BruceEckel.com
// (c) Bruce Eckel 2000
// Copyright notice in Copyright.txt
// A simple time class
#ifndef CLOCK_H
#define CLOCK_H
#include <ctime>
#include <cstring>

//! A Clock class to determine the elapsed time and to implement the delay function.
/*!
This class is composed by the classical two methods of start() and end() to determine the elapsed time in the C++ code. Furthermore the method elapsedTime() returns the time expressed in milliseconds.
Furthermore the delay(int millsec) method allows to implement a function relevant to delay.
The precision obtainable is in the order of some tenth od milliseconds.
below is shown an example of usage of this class:
\code
  double result;
  Clock C;
  // Process
  C.start();
  // loop to evaluate!!
  result=C.delay(500);
  cout<<"Time employed: "<<C.elapsedTime()<<" ms"<<endl;
\endcode

In the case a good precision of the delay time measurament is required is necessary to evaluated the delay time required for a loop cycle. For example a typical case can be the following:
\code
  double result;
  int loop;
  Clock C;
  // Process
  C.start();
   for (i=0; i<loop; i++) {
    // proces to verify
   }
  C.end();
  cout<<"Time employed: "<<C.elapsedTime()<<" ms"<<endl;

\endcode

In this case to improve the measure it is taken into account of the time required by a void for loop.
This delay time is evaluated by the loop(int loopNumber) by class Clock.

\code
  double result;
  int loop;
  Clock C;
  // Process
  C.loop(loop);
  cout<<"Time employed for void loop: "<<C.elapsedLoopTime()<<" ms"<<endl;
  C.start();
   for (i=0; i<loop; i++) {
    // proces to verify
   }
  C.end();
  cout<<"Time employed: "<<C.elapsedTime()<<" ms"<<endl; // The time obtained is more precise.
\endcode

In this case the time obtained is the elapsedTime- ElapsedTime due to the for void loop.

To reset the loop method is only necessary to call him with 0. Example C.loop(0);

*/


 class Clock {
 public:
   Clock(): startTime(0), endTime(0), loopDelay(0), loopNumber(0) {}

   //! the start() method resets the Clock internal data.
   void start(void) {
     startTime=clock();
   }

   //! the end() method stores the Clock internal data.
   void end(void) {
     endTime=clock();
   }

   //! the loop() methods evaluates the delay introduces by a for() loop statement..
   void loop(int loopNumber) {
     clock_t sTime, eTime;
     int i;
     startTime=0;
     endTime=0;
     sTime=clock();
     for (i=0; i<loopNumber; i++) {
     }
     eTime=clock();
     loopDelay=eTime-sTime; // The loopDelay has been evaluated.
   }


   //! the elapsedTime() returns the elapsed time expressed in milliseconds.
   double elapsedTime(void) const { //! the elapsed time is returned in milliseconds.
     return double (1000*(endTime-startTime-loopDelay)/CLOCKS_PER_SEC);
   }


   //! the elapsedLoopTime() returns the elapsed time expressed in milliseconds for void loop evaluation.
   double elapsedLoopTime(void) const { //! the elapsed time is returned in milliseconds.
     return double (1000*(loopDelay)/CLOCKS_PER_SEC);
   }



   //! the delay(int millsec) method evaluates a delay loop of millisec and returns the exact time elapsed. The delay time is specified in milliseconds.
   double delay(int millisec) { //! the delay time is specified in millisecond
     start();
     do
       end();
     while ( elapsedTime() < millisec);
     return elapsedTime();
   }



 private:
   clock_t startTime, endTime;
   double loopDelay; // This variables takes into account of the delay introduced by a loop
   int loopNumber;   // number of loops;

 };

#endif // CLOCK_H
