#!/bin/sh

set -e #make the script fail if a command fails
set -x #output commands to the terminal

#Test the help commands

inq help clear
inq help cell   
inq help electrons
inq help energy
inq help ground_state
inq help ions        
inq help run      
inq help theory
inq help units
inq help util
