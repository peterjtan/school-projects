using System;  
using MonoBrick.EV3; // use this to run the example on the EV3
using System.Threading;

namespace Application  
{  
    public static class Program {

        const sbyte SPEED = 100;
        const int BACKWAIT = 300;

      static void Main(string[] args)  
      {  
        try {  
            var brick = new Brick<ColorSensor,TouchSensor,GyroSensor,UltrasonicSensor>("COM3");
            brick.Connection.Close();
            brick.Connection.Open();

            brick.Sensor1.Mode = ColorMode.Color;
            brick.Sensor2.Mode = TouchMode.Boolean;
            brick.Sensor3.Mode = GyroMode.Angle;
            brick.Sensor4.Mode = UltrasonicMode.Centimeter;
            brick.Vehicle.LeftPort = MotorPort.OutA;
            brick.Vehicle.RightPort = MotorPort.OutD;
            brick.Vehicle.ReverseLeft = true; // To fix our reversed misplacement of two motors
            brick.Vehicle.ReverseRight = true;

            Console.WriteLine("Moving to white line.");
            gotoWhite(brick);

            Console.WriteLine("Move a little back and rotate.");
            brick.Vehicle.Backward(SPEED);
            Thread.Sleep(200);
            brick.Vehicle.SpinLeft(SPEED, 120); // Rotate 120 degrees

            while ( true ) // Start searching
            {
                // Press Q to quit searching and the program
                if (Console.KeyAvailable && Console.ReadKey(true).Key == ConsoleKey.Q)
                {
                    brick.Vehicle.Brake();
                    break;
                }

                Console.WriteLine("Start searching.");
                float initDegree = brick.Sensor3.Read(); // Initial angle

                switch ( search(brick) ) // Invoke the searching function
                {
                    case 'L':
                        Console.WriteLine("Should go right");
                        brick.Vehicle.SpinRight(SPEED, 60);
                        gotoWhite(brick); // Go to another place and search again
                        brick.Vehicle.Backward(SPEED);  // Move a little backward
                        Thread.Sleep(BACKWAIT);
                        brick.Vehicle.SpinLeft(SPEED, 90);
                        break;
                    
                    case 'R':
                        Console.WriteLine("Should go left");
                        brick.Vehicle.SpinLeft(SPEED, 60);
                        gotoWhite(brick);
                        brick.Vehicle.Backward(SPEED);
                        Thread.Sleep(BACKWAIT);
                        brick.Vehicle.SpinRight(SPEED, 180);
                        break;
                    
                    default:
                        Console.WriteLine("No target");
                        brick.Vehicle.SpinRight(SPEED, 60);
                        gotoWhite(brick);
                        brick.Vehicle.Backward(SPEED);
                        Thread.Sleep(BACKWAIT);
                        brick.Vehicle.SpinRight(SPEED, 180);
                        break;
                }
            }
            brick.Connection.Close();
			Console.ReadLine();
        }  
        catch(Exception e) {  
            Console.WriteLine("Error: " + e.Message);  
            Console.WriteLine("Press any key to end...");  
            Console.ReadKey();                
        }  
      }

      static char search(Brick<ColorSensor, TouchSensor, GyroSensor, UltrasonicSensor> brick)
      {
          float initDegree = brick.Sensor3.Read();
          float distance;
          brick.Vehicle.SpinLeft(SPEED);
          for (float angle = initDegree; angle > initDegree - 120; angle = brick.Sensor3.Read())
          {
              distance = brick.Sensor4.Read();
              if (distance <= 40.0)
              {
                  brick.Vehicle.Brake();
                  if (angle > initDegree - 60.0)
                  {
                      brick.Vehicle.Brake();
                      return 'R';
                  }
                  else if (angle <= initDegree - 60.0)
                  {
                      brick.Vehicle.Brake();
                      return 'L';
                  }
              }
              Thread.Sleep(100);
          }
          brick.Vehicle.Brake();
          return 'N';
      }

      static void gotoWhite(Brick<ColorSensor, TouchSensor, GyroSensor, UltrasonicSensor> brick)
      {
          bool done = false;
          int touched = 0;
          while (!done)
          {
              Thread.Sleep(100);
              brick.Vehicle.Forward(SPEED);
              if (touched == 0) // Not being touched
                  touched = brick.Sensor2.Read();
              else if (touched == 1) // Touched!
              {
                  brick.MotorD.On( -SPEED ); // Our motor is reversed
                  Thread.Sleep(500);
                  touched = 0;
              }
              if (brick.Sensor1.ReadColor() == Color.Black)
              {
                  brick.Vehicle.Brake();
                  done = true;
              }
          }
      }


    }  
} 