import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle as PltCircle
from matplotlib.patches import Rectangle as PltRectangle
import random
import math

def contains_gene(gene, genome):
    """
    Check if a gene is already present in the genome.
    
    Args:
        gene: The gene to search for
        genome: The genome to search in (list)
    
    Returns:
        bool: True if gene is in genome, False otherwise
    """
    return gene in genome

def find_first_available_position(genome):
    """
    Find the first None position in the genome.
    
    Args:
        genome: The genome to search (list with some None values)
    
    Returns:
        int: Index of first None position, or -1 if all filled
    """
    for i in range(len(genome)):
        if genome[i] is None:
            return i
    return -1

class Container:
    def __init__(self, sizeX, sizeY, capacity):
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.capacity = capacity

class Cylinder:
    def __init__(self, id, radius, weight):
        self.id = id
        self.radius = radius
        self.weight = weight
        self.x = 0
        self.y = 0
        self.placed = False

    def set_position(self, x, y):
        self.x = x
        self.y = y
        self.placed = True

    def distance_to(self, other):
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def overlaps(self, other):
        distance = self.distance_to(other)
        return distance < (self.radius + other.radius - 0.01)

    def distance_from_origin(self):
        return np.sqrt(self.x**2 + self.y**2)

class Individual:
    def __init__(self, num_genes, container, cylinders):
        self.num_genes = num_genes
        self.container = container
        self.cylinders = []
        self.centreOfMassX = 0
        self.centreOfMassY = 0
        for cylinder in cylinders:
            self.cylinders.append(Cylinder(cylinder.id, cylinder.radius, cylinder.weight))

        self.genes = [cylinders[i].id for i in range(num_genes)]
        random.shuffle(self.genes)
        
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self):

        self.ordered_place()
        self.calculate_centre_of_mass()

        placed = [c for c in self.cylinders if c.placed]
        totalWeight = 0
        for c in self.cylinders:
            totalWeight = totalWeight + c.weight

        if totalWeight > container.capacity:
            return 100
        
        if self.centreOfMassX > (self.container.sizeX / 2) or self.centreOfMassX < -(self.container.sizeX/2) or self.centreOfMassY > (self.container.sizeY/2) or self.centreOfMassY < -(self.container.sizeY/2):
            return 100

        if len(placed) < len(self.cylinders):
            return (10 * (len(self.cylinders) - len(placed))) + self.compute_boundary(self.cylinders, 0, 0)

        return self.compute_boundary(self.cylinders, 0, 0)

    def ordered_place(self):
        valid = False
        while valid == False:
            for i in range(len(self.genes)):
                open_points = self.find_open_points(self.cylinders[self.genes[i] - 1])
                if open_points:
                    best_position = max(open_points, key=lambda p: p[2])
                    self.cylinders[self.genes[i] - 1].set_position(best_position[0], best_position[1])
                placed = [c for c in self.cylinders if c.placed]
            
            #if len(placed) == len(self.cylinders):
            valid = True
            
    
    def compute_boundary(self, cylinders, center_x, center_y):
        max_distance = 0

        for cylinder in cylinders:
            dx = cylinder.x - center_x
            dy = cylinder.y - center_y
            dist_to_center = math.sqrt(dx*dx + dy*dy)
            dist_to_edge = dist_to_center + cylinder.radius

            if dist_to_edge > max_distance:
                max_distance = dist_to_edge

        return max_distance
    
    def calculate_centre_of_mass(self):
        totalCylinderMassX = 0
        totalCylinderMassY = 0
        totalCylinderMass = 0
        for i in range(len(self.cylinders)):
            totalCylinderMass = totalCylinderMass + (self.cylinders[i].weight)
            totalCylinderMassX = totalCylinderMassX + (self.cylinders[i].weight * self.cylinders[i].x)
            totalCylinderMassY = totalCylinderMassY + (self.cylinders[i].weight * self.cylinders[i].y)
        self.centreOfMassX = totalCylinderMassX / totalCylinderMass
        self.centreOfMassY = totalCylinderMassY / totalCylinderMass

    def find_open_points(self, newCylinder):
        open_points = []
        placed = [c for c in self.cylinders if c.placed]

        #if len(placed) == 0:
            #return [(-(self.container.sizeX / 2) + newCylinder.radius, (self.container.sizeY / 2) - newCylinder.radius, 0)]
        
        if len(placed) >= 0:
            """amount = 0
            for c in placed:
                if amount <= 1000:
                    distance = c.radius + newCylinder.radius
                    for angle in np.linspace(0, 2*np.pi, 36, endpoint=False):
                        x = c.x + distance * np.cos(angle)
                        y = c.y + distance * np.sin(angle)
                        temp_cylinder = Cylinder(newCylinder.id, newCylinder.radius, newCylinder.weight)
                        temp_cylinder.set_position(x, y)

                        valid = True
                        for other in placed:
                            if temp_cylinder.overlaps(other):
                                valid = False
                        self.calculate_centre_of_mass()
                        #print(str(self.centreOfMassX) + " " + str(self.centreOfMassY))
                        if x + newCylinder.radius > (self.container.sizeX / 2) or y + newCylinder.radius > (self.centreOfMassY/2) or x - newCylinder.radius < -(self.container.sizeX/2) or y - newCylinder.radius < -(self.container.sizeY/2):
                            valid = False
                        if self.centreOfMassX > (self.container.sizeX / 2) or self.centreOfMassX < -(self.container.sizeX/2) or self.centreOfMassY > (self.container.sizeY/2) or self.centreOfMassY < -(self.container.sizeY/2):
                            valid = False
                    
                        if valid:
                            temp_cylinders = placed
                            temp_cylinders.append(temp_cylinder)
                            boundry = self.compute_boundary(temp_cylinders, 0, 0)
                            open_points.append((x, y, boundry))
                            amount += 1"""

            """maxLoops = 100
            for i in range(0, maxLoops):
                x = random.uniform(-(self.container.sizeX/2) + newCylinder.radius, (self.container.sizeX/2) - newCylinder.radius)
                y = random.uniform(-(self.container.sizeY/2) + newCylinder.radius, (self.container.sizeY/2) - newCylinder.radius)
                temp_cylinder = Cylinder(newCylinder.id, newCylinder.radius, newCylinder.weight)
                temp_cylinder.set_position(x, y)

                valid = True
                for other in placed:
                    if temp_cylinder.overlaps(other):
                        valid = False
                self.calculate_centre_of_mass()
                #print(str(self.centreOfMassX) + " " + str(self.centreOfMassY))
                if self.centreOfMassX > (self.container.sizeX / 2) or self.centreOfMassX < -(self.container.sizeX/2) or self.centreOfMassY > (self.container.sizeY/2) or self.centreOfMassY < -(self.container.sizeY/2):
                    valid = False
                
                if valid:
                    temp_cylinders = placed
                    temp_cylinders.append(temp_cylinder)
                    boundry = self.compute_boundary(temp_cylinders, 0, 0)
                    open_points.append((x, y, boundry))"""
            
            for x in range(int(container.sizeX * 2)):
                for y in range(int(container.sizeY * 2)):
                    temp_cylinder = Cylinder(newCylinder.id, newCylinder.radius, newCylinder.weight)
                    temp_cylinder.set_position((x / 2) - (container.sizeX / 2), (y / 2) - (container.sizeY / 2))

                    valid = True
                    for other in placed:
                        if temp_cylinder.overlaps(other):
                            valid = False
    
                    if (temp_cylinder.x + temp_cylinder.radius) > (container.sizeX / 2) or (temp_cylinder.x - temp_cylinder.radius) < -(container.sizeX / 2) or (temp_cylinder.y + temp_cylinder.radius) > (container.sizeY / 2) or (temp_cylinder.y - temp_cylinder.radius) < -(container.sizeY / 2):
                        valid = False
                    
                    if valid:
                        #dist_from_origin = np.sqrt(x - (container.sizeX / 2)**2 + x - (container.sizeY / 2)**2)
                        #rand = random.randrange(0, 100)
                        open_points.append((temp_cylinder.x, temp_cylinder.y, y))
                        return open_points
                        

        """if len(placed) == 1:
            c1 = placed[0]
            distance = c1.radius + newCylinder.radius
            for angle in np.linspace(0, 2*np.pi, 36, endpoint=False):
                x = c1.x + distance * np.cos(angle)
                y = c1.y + distance * np.sin(angle)
                dist_from_origin = np.sqrt(x**2 + y**2)
                open_points.append((x, y, dist_from_origin))
            return open_points
        
        for i in range(len(placed)):
            for j in range(i+1, len(placed)):
                c1 = placed[i]
                c2 = placed[j]

                positions = self.find_tangent_positions(c1, c2, newCylinder)
                for x, y in positions:
                    temp_cylinder = Cylinder(newCylinder.id, newCylinder.radius, newCylinder.weight)
                    temp_cylinder.set_position(x, y)

                    valid = True
                    for other in placed:
                        if other != c1 and other != c2:
                            if temp_cylinder.overlaps(other):
                                valid = False
                                break
                    if x + temp_cylinder.radius > (self.container.sizeX / 2) or y + temp_cylinder.radius > (self.container.sizeY / 2) or x - temp_cylinder.radius < -(self.container.sizeX / 2) or y - temp_cylinder.radius < -(self.container.sizeY / 2):
                        valid = False
                    
                    if valid:
                        dist_from_origin = np.sqrt(x**2 + y**2)
                        open_points.append((x, y, dist_from_origin))"""

        return open_points
    
    def draw(self, title="Cylinder Placement"):
        fig, ax = plt.subplots(figsize=(10, 10))

        container_rectangle = PltRectangle((-container.sizeX * 5, -container.sizeY * 5), container.sizeX * 10, container.sizeY * 10,
                                           fill=False, edgecolor='#F4BA02',
                                           linewidth=2, linestyle='--', label='Containing rectangle')
        ax.add_patch(container_rectangle)

        for cylinder in self.cylinders:
            if cylinder.placed:
                cylinder_patch = PltCircle((cylinder.x * 10, cylinder.y * 10), cylinder.radius * 10,
                                           fill=False, edgecolor='#99D9DD', linewidth=2)
                ax.add_patch(cylinder_patch)
                ax.plot(cylinder.x * 10, cylinder.y * 10, 'o', color='#99D9DD', markersize=6)
                ax.text(cylinder.x * 10, cylinder.y * 10, f'{int(cylinder.radius)}', ha='center', va='center', color='#F7F8F9', fontsize=9)

        ax.plot(self.centreOfMassX * 10, self.centreOfMassY * 10, 'x', color='#F4BA02', markersize=12, markeredgewidth=3, label='Centre of Mass')
        print(str(self.centreOfMassX) + " " + str(self.centreOfMassY))

        ax.set_aspect('equal')
        ax.set_xlim(-300, 300)
        ax.set_ylim(-250, 250)

        ax.grid(True, alpha=0.3, color='#F7F8F9')
        ax.set_facecolor('#01364C')
        fig.patch.set_facecolor('#01364C')
        ax.tick_params(colors='#F7F8F9')
        for spine in ax.spines.values():
            spine.set_color('#F7F8F9')
        
        ax.set_title(f"{title}\nFitness: {self.fitness:.2f}", 
                    color='#F7F8F9', fontsize=14, pad=20, weight='bold')
        ax.legend(loc='upper right', facecolor='#01364C', edgecolor='#F7F8F9', 
                 labelcolor='#F7F8F9', framealpha=0.9)

        plt.show()
    
    def find_tangent_positions(self, c1, c2, newCylinder):
        r1 = c1.radius + newCylinder.radius
        r2 = c2.radius + newCylinder.radius
        d = c1.distance_to(c2)

        if d > r1 + r2 or d < abs(r1 - r2) or d == 0:
            return []
        
        a = (r1**2 - r2**2 + d**2) / (2*d)
        h = np.sqrt(max(0, r1**2 - a**2))

        px = c1.x + a * (c2.x - c1.x) / d
        py = c1.y + a * (c2.y - c1.y) / d

        positions = []
        if h > 0.01:
            positions.append((
                px + h * (c2.y - c1.y) / d,
                py - h * (c2.x - c1.x) / d
            ))
            positions.append((
                px - h * (c2.y - c1.y) / d,
                py + h * (c2.x - c1.x) / d
            ))
        else:
            positions.append((px, py))
        
        return positions
    
    def mutate(self):

        for i in range(random.randrange(1, 5)):
            r1 = random.randrange(len(self.genes))
            r2 = random.randrange(len(self.genes))
            g1 = self.genes[r1]
            g2 = self.genes[r2]

            self.genes[r1] = g2
            self.genes[r2] = g1

        self.fitness = self.calculate_fitness()


class Population:
    def __init__(self, pop_size, num_genes, container, cylinders):
        self.pop_size = pop_size
        self.num_genes = num_genes
        self.container = container
        self.cylinders = cylinders

        self.individuals = [
            Individual(num_genes, container, cylinders) for _ in range(pop_size)
        ]

    def evolve(self, elitism=True):
        new_individuals = []

        if elitism:
            new_individuals.append(self.get_best_individual())

        while len(new_individuals) < self.pop_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()

            offspringGenes = self.ordered_crossover(parent1, parent2)
            
            offspring = parent1

            offspring.genes = offspringGenes

            offspring.mutate()

            new_individuals.append(offspring)

        self.individuals = new_individuals

    def ordered_crossover(self, parent1, parent2):
        genome_length = len(parent1.genes)

        child = [None] * genome_length

        start = random.randint(0, genome_length - 2)
        end = random.randint (start + 1, genome_length)

        for i in range(start, end):
            child[i] = parent1.genes[i]

        current_pos = end % genome_length

        for i in range(genome_length):
            parent2_pos = (end + i) % genome_length
            gene = parent2.genes[parent2_pos]

            if not contains_gene(gene, child):
                insert_pos = find_first_available_position(child)
                child[insert_pos] = gene
        
        return child

    def tournament_selection(self, tournament_size=5):
        tournament = random.sample(self.individuals, tournament_size)

        return min(tournament, key=lambda ind: ind.fitness)

    def get_best_individual(self):
        return min(self.individuals, key=lambda ind: ind.fitness)

    def get_stats(self):
        fitnesses = [ind.fitness for ind in self.individuals]
        return {
            'best': max(fitnesses),
            'average': sum(fitnesses) / len(fitnesses),
            'worst': min(fitnesses)
        }

container=Container(15.0, 12.0, 200.0)
cylinders=[
            Cylinder(1, 3.5, 25.0),
            Cylinder(2, 3.0, 20.0),
            Cylinder(3, 2.5, 18.0),
            Cylinder(4, 2.5, 18.0),
            Cylinder(5, 2.0, 15.0)
        ]

pop = Population(
    pop_size=500,
    num_genes=len(cylinders),
    container=container,
    cylinders=cylinders
)

num_generations = 10
for gen in range(num_generations):
    pop.evolve(elitism=True)

    if gen % 2 == 0:
        stats = pop.get_stats()
        print(f"Generation {gen}: Best={stats['best']:.2f}, Avg={stats['average']:.2f}")

best = pop.get_best_individual()
print(str(best.genes))
stats = pop.get_stats()
print(f"Generation {gen}: Best={stats['best']:.2f}, Avg={stats['average']:.2f}")
best.draw()